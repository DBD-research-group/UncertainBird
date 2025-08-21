# /workspace/projects/uncertainbird/configs/experiment/Wrappers/mc_predictor.py

from __future__ import annotations
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import Any, Dict, List, Tuple, Optional, Union
from birdset.modules.metrics.multilabel import MultilabelECEMacro, TopKAccuracy

# torchmetrics (only imported if we compute metrics)
try:
    from torchmetrics.classification import (
        MultilabelAveragePrecision,
        MultilabelAUROC,
        MulticlassAUROC,
        MulticlassAccuracy,
    )
    _HAS_TM = True
except Exception:
    _HAS_TM = False

__all__ = ("MCDropoutPredictor", "mc_predict")

# ------------------------- utils ---------------------------------------------


def _is_dropout_like(m: nn.Module) -> bool:
    name = m.__class__.__name__.lower()
    return (
        isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout))
        or "dropout" in name
    )


def _freeze_norms(m: nn.Module):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        m.eval()


def _stack_if_list(x):
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(t, torch.Tensor) for t in x):
        return torch.stack(list(x), dim=0)
    return x


def _maybe_extract_tensor(v):
    """Return a Tensor if v looks tensor-like; handle lists/tuples and nested dicts."""
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(t, torch.Tensor) for t in v):
        return torch.stack(list(v), dim=0)
    if isinstance(v, dict):
        for inner_k in ("tensor", "data", "input", "values"):
            if inner_k in v:
                return _maybe_extract_tensor(v[inner_k])
    return None


def _first_tensor_like(d: dict, exclude_keys=()):
    """Pick the first tensor-like value from a dict, skipping excluded keys."""
    for k, v in d.items():
        if k in exclude_keys:
            continue
        tv = _maybe_extract_tensor(v)
        if isinstance(tv, torch.Tensor):
            return tv, k
    return None, None


def _flatten_outputs(outputs):
    flat = []
    for elem in outputs:
        flat.extend(elem if isinstance(elem, list) else [elem])
    return flat


def _resolve_dataloaders(datamodule=None, dataloader=None):
    """
    Accept either a ready dataloader, or a datamodule.
    If a datamodule is given, try predict -> test -> val (calling setup when available).
    """
    if dataloader is not None:
        return dataloader
    if datamodule is None:
        raise ValueError("Provide either `dataloader` or `datamodule`.")

    # predict
    if hasattr(datamodule, "setup"):
        try:
            datamodule.setup("predict")
        except TypeError:
            datamodule.setup()
        except Exception:
            pass
    if hasattr(datamodule, "predict_dataloader"):
        try:
            dl = datamodule.predict_dataloader()
            if dl is not None:
                return dl
        except Exception:
            pass

    # test
    if hasattr(datamodule, "setup"):
        try:
            datamodule.setup("test")
        except TypeError:
            datamodule.setup()
        except Exception:
            pass
    if hasattr(datamodule, "test_dataloader"):
        dl = datamodule.test_dataloader()
        if dl is not None:
            return dl

    # val
    if hasattr(datamodule, "setup"):
        try:
            datamodule.setup("validate")
        except TypeError:
            datamodule.setup()
        except Exception:
            pass
    if hasattr(datamodule, "val_dataloader"):
        dl = datamodule.val_dataloader()
        if dl is not None:
            return dl

    raise RuntimeError("No dataloader found. Implement predict/test/val loaders or pass one explicitly.")


# --------------------- wrapper module ----------------------------------------


class MCDropoutPredictor(LightningModule):
    """
    Lightning wrapper that:
      - enables MC-Dropout (Dropout active, BN frozen)
      - runs T stochastic passes and aggregates
      - (optional) aggregates metrics across the *predict* epoch
    """

    def __init__(
        self,
        base_model: Union[LightningModule, nn.Module],
        T: int = 10,
        threshold: float = 0.5,
        task: str = "multilabel",  # or "multiclass"
        num_labels: Optional[int] = None,
        compute_metrics: bool = True,
        average: str = "macro",
        topk_eval: Tuple[int, int] = (1, 3),
    ):
        super().__init__()
        self.base_model = base_model
        self.T = int(T)
        self.threshold = float(threshold)
        self.task = task
        self.num_labels = num_labels
        self.compute_metrics = compute_metrics and _HAS_TM
        self.average = average
        self.topk_eval = topk_eval

        # buffers for metrics
        self._probs: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

        # store final metrics here
        self.final_metrics: Dict[str, float] = {}

        # track how many dropout modules we see
        self._mc_drop_count: int = 0

    # ---- MC mode helpers ----

    def _activate_mc_dropout(self):
        """Enable stochastic layers regardless of Lightning’s eval() calls."""
        if getattr(self, "_mc_activated", False):
            return

        # If user attached EAT hooks, turn them on
        try:
            from eat_dropout_hooks import set_eat_mc_mode  # path must be importable
            set_eat_mc_mode(getattr(self.base_model, "model", self.base_model), True, freeze_batchnorm=True)
        except Exception:
            pass

        # Turn ON training globally so functional dropout respects it
        self.base_model.train()

        # Freeze BN stats
        self.base_model.apply(_freeze_norms)

        # Count dropout-like modules (for debug)
        self._mc_drop_count = sum(1 for m in self.base_model.modules() if _is_dropout_like(m))
        self.print(f"[MC] activated. Dropout-like modules: {self._mc_drop_count}; base_model.training={self.base_model.training}")
        self._mc_activated = True

    def _deactivate_mc_dropout(self):
        try:
            from eat_dropout_hooks import set_eat_mc_mode
            set_eat_mc_mode(getattr(self.base_model, "model", self.base_model), False, freeze_batchnorm=True)
        except Exception:
            pass
        self.base_model.eval()
        self._mc_activated = False

    # ---- I/O helpers ----

    def _get_xy(self, batch):
        """
        Robust extractor for (x, y) from dict/tuple/list batches.
        Handles HF keys like 'input_values', 'pixel_values', 'input_ids', etc.
        Falls back to the first tensor-like entry if known keys are missing/None.
        """
        x = y = None

        if isinstance(batch, dict):
            input_key_candidates = (
                "input_values",   # HF audio
                "pixel_values",   # HF vision
                "input_ids",      # HF text
                "inputs_embeds",
                "image", "images", "img",
                "x", "input", "inputs",
                "features", "data",
            )
            label_key_candidates = ("labels", "label", "y", "targets", "target")

            # Try explicit input keys; skip None/invalid values
            for k in input_key_candidates:
                if k in batch:
                    tv = _maybe_extract_tensor(batch[k])
                    if tv is not None:
                        x = tv
                        break

            # Labels
            for k in label_key_candidates:
                if k in batch:
                    ty = _maybe_extract_tensor(batch[k])
                    if ty is not None:
                        y = ty
                        break

            # Fallback: pick the first tensor-like value (exclude label keys if possible)
            if x is None:
                x, _ = _first_tensor_like(batch, exclude_keys=label_key_candidates)

        elif isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("Empty batch")
            if isinstance(batch[0], dict):
                x, y = self._get_xy(batch[0])
            else:
                x = _maybe_extract_tensor(batch[0]) or _stack_if_list(batch[0])
                y = _maybe_extract_tensor(batch[1]) if len(batch) > 1 else None
        else:
            x = _maybe_extract_tensor(batch) or batch  # raw tensor

        if x is None:
            keys = list(batch.keys()) if isinstance(batch, dict) else type(batch)
            raise KeyError(f"Could not find input tensor in batch. Available: {keys}")

        # Common shape fixes
        if isinstance(x, torch.Tensor):
            # audio: [B, L] -> [B, 1, L]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            # vision: channel-last -> channel-first
            if x.dim() == 4 and x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2).contiguous()
            x = x.to(self.device, non_blocking=True)

        if isinstance(y, torch.Tensor):
            y = y.to(self.device, non_blocking=True)

        return x, y

    # ---- Lightning hooks ----

    def on_predict_start(self):
        # Determine number of labels if possible
        if self.num_labels is None:
            # try inner model attr
            self.num_labels = getattr(getattr(self.base_model, "model", self.base_model), "num_classes", None)
        if self.num_labels is None:
            # try LightningModule hparams
            self.num_labels = getattr(getattr(self.base_model, "hparams", {}), "num_classes", None)
        if self.num_labels is None:
            self.print("[MC] Warning: num_labels unknown; metrics that need it will be skipped.")

        # init metric accumulators
        self._probs.clear()
        self._targets.clear()
        self.final_metrics = {}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._activate_mc_dropout()
        x, y = self._get_xy(batch)

        probs_T = []
        with torch.no_grad():
            for _ in range(self.T):
                logits = self.base_model(x)  # [B, C]
                if self.task == "multilabel":
                    probs_T.append(torch.sigmoid(logits))
                else:  # multiclass
                    probs_T.append(torch.softmax(logits, dim=-1))

        probs_T = torch.stack(probs_T, dim=0)  # [T, B, C]
        p_mean = probs_T.mean(dim=0)           # [B, C]
        p_var = probs_T.var(dim=0, unbiased=False)

        # predictions
        if self.task == "multilabel":
            y_hat = (p_mean > self.threshold).int()
        else:
            y_hat = p_mean.argmax(dim=-1)  # [B]

        # collect for metrics
        if self.compute_metrics and (y is not None):
            # For multiclass, ensure y is [B] int
            if self.task == "multiclass" and y.dim() > 1:
                y = y.argmax(dim=-1)
            self._probs.append(p_mean.detach().cpu())
            self._targets.append(y.detach().cpu())

        return {"p_mean": p_mean, "p_var": p_var, "y_hat": y_hat, "y": y}

    def on_predict_epoch_end(self):
        # compute metrics if requested
        if not (self.compute_metrics and len(self._probs) > 0 and self.num_labels):
            return

        probs = torch.cat(self._probs, dim=0)     # [N, C]
        targets = torch.cat(self._targets, dim=0) # [N, C] (multilabel) or [N] (multiclass)

        metrics: Dict[str, float] = {}

        if self.task == "multilabel":
            probs_t = probs.clone()
            targ_t = targets.clone().to(torch.int64)
            probs_t = probs_t.float()
            
            print("probs_t:", probs_t[:5])   # print first 5 samples
            print("targ_t:", targ_t[:5])


            # mAP & AUROC (macro)
            m1 = MultilabelAveragePrecision(num_labels=self.num_labels, average=self.average)
            m2 = MultilabelAUROC(num_labels=self.num_labels, average=self.average)
            m3 = MultilabelECEMacro(num_labels=self.num_labels, bins=10, threshold=0.5)
            

            metrics["mc/mAP_macro"] = float(m1(probs_t, targ_t).item())
            metrics["mc/AUROC_macro"] = float(m2(probs_t, targ_t).item())
            metrics["mc/mECE"] = float(m3(probs_t, targ_t).item())

            # Top-k multilabel accuracy (hit if any true class in top-k)
            with torch.no_grad():
                B, C = probs_t.shape
                for k in self.topk_eval:
                    k_eff = min(k, C)
                    topk_idx = probs_t.topk(k_eff, dim=1).indices  # [B, k]
                    true_mask = targ_t.bool()
                    hit = true_mask.gather(1, topk_idx).any(dim=1)  # [B]
                    metrics[f"mc/T{k}Accuracy"] = float(hit.float().mean().item())

        else:  # multiclass
            probs_t = probs.clone()
            targ_t = targets.clone().to(torch.int64)

            # AUROC macro (OvR)
            m_auc = MulticlassAUROC(num_classes=self.num_labels, average=self.average)
            metrics["mc/AUROC_macro"] = float(m_auc(probs_t, targ_t).item())

            # Top-1 / Top-3 accuracy
            for k in self.topk_eval:
                k_eff = min(k, self.num_labels)
                m_acc = MulticlassAccuracy(num_classes=self.num_labels, top_k=k_eff)
                metrics[f"mc/Top{k_eff}Acc"] = float(m_acc(probs_t, targ_t).item())

        self.final_metrics = metrics

        # log to logger, if available (e.g., W&B)
        if getattr(self.trainer, "logger", None) and hasattr(self.trainer.logger, "experiment"):
            try:
                self.trainer.logger.experiment.log(metrics)
            except Exception:
                pass

        # turn MC off after predict
        self._deactivate_mc_dropout()


# ---------------------- public helper ----------------------------------------


def mc_predict(
    trainer,
    base_model,
    datamodule=None,
    dataloader=None,
    T: int = 10,
    threshold: float = 0.5,
    task: str = "multilabel",
    num_labels: Optional[int] = None,
    compute_metrics: bool = True,
    ckpt_path: Optional[str] = None,
):
    """
    Run MC-Dropout predict using the already-loaded base_model.
    Accepts either a datamodule or a ready dataloader.

    Returns:
        {
          "p_mean": Tensor [N, C],
          "p_var":  Tensor [N, C],
          "y_hat":  Tensor [N, C] (multilabel) or [N] (multiclass),
          "y":      Tensor or None,
          "metrics": Dict[str, float]   # if computed
        }
    """
    wrapper = MCDropoutPredictor(
        base_model=base_model,
        T=T,
        threshold=threshold,
        task=task,
        num_labels=num_labels,
        compute_metrics=compute_metrics,
    )
    dl = _resolve_dataloaders(datamodule=datamodule, dataloader=dataloader)
    outputs = trainer.predict(model=wrapper, dataloaders=dl, ckpt_path=ckpt_path)

    flat = _flatten_outputs(outputs)
    p_mean = torch.cat([o["p_mean"] for o in flat], dim=0).detach().cpu()
    p_var = torch.cat([o["p_var"] for o in flat], dim=0).detach().cpu()

    # labels (may be None)
    y = flat[0].get("y", None)
    if y is not None:
        y = torch.cat([o["y"] for o in flat], dim=0).detach().cpu()

    # predictions
    if task == "multilabel":
        y_hat = (p_mean > threshold).to(torch.int32)
    else:
        y_hat = p_mean.argmax(dim=-1)

    result = {"p_mean": p_mean, "p_var": p_var, "y_hat": y_hat, "y": y}

    # add metrics if present
    if getattr(wrapper, "final_metrics", None):
        result["metrics"] = wrapper.final_metrics

    return result
