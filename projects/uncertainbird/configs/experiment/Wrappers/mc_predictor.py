# /workspace/projects/uncertainbird/configs/experiment/Wrappers/mc_predictor.py
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

__all__ = ("MCDropoutPredictor", "mc_predict")

# ------------------------- utils ---------------------------------------------

def _enable_dropout_only(m: nn.Module):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.train()



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
        try: datamodule.setup("predict")
        except TypeError: datamodule.setup()
        except Exception: pass
    if hasattr(datamodule, "predict_dataloader"):
        try:
            dl = datamodule.predict_dataloader()
            if dl is not None: return dl
        except Exception:
            pass

    # test
    if hasattr(datamodule, "setup"):
        try: datamodule.setup("test")
        except TypeError: datamodule.setup()
        except Exception: pass
    if hasattr(datamodule, "test_dataloader"):
        dl = datamodule.test_dataloader()
        if dl is not None: return dl

    # val
    if hasattr(datamodule, "setup"):
        try: datamodule.setup("validate")
        except TypeError: datamodule.setup()
        except Exception: pass
    if hasattr(datamodule, "val_dataloader"):
        dl = datamodule.val_dataloader()
        if dl is not None: return dl

    raise RuntimeError("No dataloader found. Implement predict/test/val loaders or pass one explicitly.")

# --------------------- wrapper module ----------------------------------------

class MCDropoutPredictor(LightningModule):
    def __init__(self, base_model: LightningModule, T: int = 10, threshold: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.T = int(T)
        self.threshold = float(threshold)

    def on_predict_start(self):
        # Turn on training globally so F.dropout(..., training=self.training) actually fires
        self.base_model.train()

        # Freeze normalization layers to eval (no running stats updates / BN stochasticity)
        NORM_TYPES = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
            # add more if your model uses them and you want them frozen:
            # nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
            # nn.GroupNorm, nn.LayerNorm,
        )
    def _activate_mc_dropout(self):
        """Enable stochastic layers regardless of Lightning's eval() calls."""
        if getattr(self, "_mc_activated", False):
            return
        # turn ON training globally so functional dropout respects it
        self.base_model.train()

        # freeze normalization layers to eval so BN stats don't update
        NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        def _freeze_norms(m: nn.Module):
            if isinstance(m, NORM_TYPES):
                m.eval()
        self.base_model.apply(_freeze_norms)

        # debug info
        def _is_dropout_like(m):
            name = m.__class__.__name__.lower()
            return isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)) or "dropout" in name
        self._mc_drop_count = sum(1 for m in self.base_model.modules() if _is_dropout_like(m))
        self.print(f"[MC] activated dropout; modules found: {self._mc_drop_count}; base_model.training={self.base_model.training}")
        self._mc_activated = True

        def _freeze_norms(m: nn.Module):
            if isinstance(m, NORM_TYPES):
                m.eval()

        self.base_model.apply(_freeze_norms)

        # (Optional) If you also want to ensure classic Dropout modules are active, nothing else is needed;
        # since the model is in train(), Dropout modules are already "on".
        # But you can still count/log them if you want:
        # count = sum(1 for m in self.base_model.modules()
        #             if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)))
        # self.print(f"[MC] enabled dropout modules: {count}")

    def forward(self, x):
        return self.base_model(x)

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
                x, picked = _first_tensor_like(batch, exclude_keys=label_key_candidates)
                # As a last resort, if we accidentally picked labels as x,
                # try to find a different tensor for y later.

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

        # Auto-convert channel-last [B,H,W,C] -> [B,C,H,W] when obvious
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            B, A, B2, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            # if channels are last and look like 1/3
            if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2).contiguous()

        # Move to correct device
        if isinstance(x, torch.Tensor):
            x = x.to(self.device, non_blocking=True)

        return x, y

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._activate_mc_dropout()
        self.print(f"[MC] T={self.T}")
        x, y = self._get_xy(batch)

        probs_T = []
        with torch.no_grad():
            for _ in range(self.T):
                logits = self.base_model(x)            # [B, C]
                probs_T.append(torch.sigmoid(logits))  # [B, C]

        probs_T = torch.stack(probs_T, dim=0)          # [T, B, C]
        p_mean  = probs_T.mean(dim=0)                  # [B, C]
        p_var   = probs_T.var(dim=0, unbiased=True)    # [B, C]
        y_hat   = (p_mean > self.threshold).int()      # [B, C]

        return {"p_mean": p_mean, "p_var": p_var, "y_hat": y_hat, "y": y}

# ---------------------- public helper ----------------------------------------

def mc_predict(trainer, base_model, datamodule=None, dataloader=None,
               T: int = 10, threshold: float = 0.5, ckpt_path: str | None = None):
    """
    Run MC-Dropout predict using the already-loaded base_model.
    Accepts either a datamodule or a ready dataloader.
    """
    wrapper = MCDropoutPredictor(base_model, T=T, threshold=threshold)
    dls = _resolve_dataloaders(datamodule=datamodule, dataloader=dataloader)
    outputs = trainer.predict(model=wrapper, dataloaders=dls)

    flat = _flatten_outputs(outputs)

    p_mean = torch.cat([o["p_mean"] for o in flat], dim=0)
    p_var  = torch.cat([o["p_var"]  for o in flat], dim=0)

    first = flat[0]
    y = first.get("y", None)
    if y is not None:
        y = torch.cat([o["y"] for o in flat], dim=0)

    y_hat = (p_mean > threshold).int()
    return {"p_mean": p_mean, "p_var": p_var, "y_hat": y_hat, "y": y}
