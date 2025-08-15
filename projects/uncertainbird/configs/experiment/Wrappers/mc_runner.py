# /workspace/projects/uncertainbird/configs/experiment/Wrappers/mc_runner.py

from __future__ import annotations
import torch
from typing import Optional, Dict, Any, List

# Import the wrapper LightningModule
from mc_predictor import MCDropoutPredictor

__all__ = ("mc_predict_from_ckpt",)

def _flatten_outputs(outputs: List[Any]) -> List[Dict[str, torch.Tensor]]:
    """
    Lightning's predict() returns a list of per-dataloader outputs.
    Each item can itself be a list of per-batch dicts.
    This flattens to a simple list of dicts (one per batch).
    """
    flat: List[Dict[str, torch.Tensor]] = []
    for elem in outputs:
        if isinstance(elem, list):
            for sub in elem:
                if isinstance(sub, list):
                    flat.extend(sub)
                else:
                    flat.append(sub)
        else:
            flat.append(elem)
    return flat

def _safe_load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """
    Load a Lightning checkpoint robustly under PyTorch 2.6+.
    Tries safe (weights_only) load with OmegaConf allowlist; falls back to full unpickle.
    """
    try:
        # Allowlist common OmegaConf types found in Lightning ckpts
        import torch.serialization as ser
        from omegaconf.dictconfig import DictConfig
        try:
            # ContainerMetadata moved packages across OmegaConf versions; handle if present
            from omegaconf.base import ContainerMetadata  # type: ignore
            allowed = [DictConfig, ContainerMetadata]
        except Exception:
            allowed = [DictConfig]

        with ser.safe_globals(allowed):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ckpt
    except Exception as e:
        print(f"[mc_runner] Safe load failed ({e}); falling back to weights_only=False")
        # Only do this if you trust the checkpoint source.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt

def mc_predict_from_ckpt(
    trainer,
    base_model,
    datamodule=None,
    dataloader=None,
    ckpt_path: Optional[str] = None,
    T: int = 10,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Run MC-Dropout prediction with an optional checkpoint load.

    Args:
        trainer: Lightning Trainer
        base_model: your trained LightningModule (will receive loaded weights if ckpt_path is given)
        datamodule: LightningDataModule to pass to trainer.predict (optional)
        dataloader: alternative DataLoader(s) if not using a datamodule (optional)
        ckpt_path: checkpoint to load (optional). If None, uses in-memory weights.
        T: number of stochastic passes with dropout enabled
        threshold: decision threshold applied AFTER probability averaging

    Returns:
        dict with:
          - p_mean: [N, C] averaged probabilities
          - p_var : [N, C] sample variance across T passes
          - y_hat : [N, C] thresholded predictions (ints)
          - y     : [N, C] labels if available, else None
    """
    # Optionally load checkpoint weights into the provided model
    if ckpt_path:
        ckpt = _safe_load_checkpoint(ckpt_path)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        base_model.load_state_dict(state, strict=True)

    # Wrap with MC-Dropout predictor
    wrapper = MCDropoutPredictor(base_model, T=T, threshold=threshold)

    # Run predict
    if datamodule is not None:
        outputs = trainer.predict(model=wrapper, datamodule=datamodule)
    else:
        outputs = trainer.predict(model=wrapper, dataloaders=dataloader)

    flat = _flatten_outputs(outputs)

    # Stack/concat results
    p_mean = torch.cat([o["p_mean"] for o in flat], dim=0)
    p_var  = torch.cat([o["p_var"]  for o in flat], dim=0)

    # y is optional
    first = flat[0]
    y = first.get("y", None)
    if y is not None:
        y = torch.cat([o["y"] for o in flat], dim=0)

    y_hat = (p_mean > threshold).int()

    return {"p_mean": p_mean, "p_var": p_var, "y_hat": y_hat, "y": y}

# --- Optional backwards-compatible alias (use at your own risk to avoid naming clashes) ---
# If you *really* want to import mc_predict from this module, uncomment the alias below.
# But it's safer to import `mc_predict_from_ckpt` to avoid confusion with mc_predictor.mc_predict.
# mc_predict = mc_predict_from_ckpt
