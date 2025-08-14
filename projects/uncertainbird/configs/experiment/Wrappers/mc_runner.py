# mc_runner.py
import torch
from mc_predictor import MCDropoutPredictor

def mc_predict(trainer, base_model, datamodule=None, dataloader=None,
               ckpt_path=None, T: int = 10, threshold: float = 0.5):
    # --- robust checkpoint loading for PyTorch 2.6+ ---
    if ckpt_path:
        try:
            # Try safe loading with an allowlist for OmegaConf
            from omegaconf.dictconfig import DictConfig
            import torch.serialization as ser
            with ser.safe_globals([DictConfig]):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            # Fall back to full unpickling (only if you trust your checkpoint)
            print(f"[mc_predict] Safe load failed ({e}); falling back to weights_only=False")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        base_model.load_state_dict(state, strict=True)

    wrapper = MCDropoutPredictor(base_model, T=T, threshold=threshold)

    # Use predict so we don't touch your test loop at all
    out = trainer.predict(model=wrapper, datamodule=datamodule)

    # `out` is a list of batches (or list of list if multiple loaders)
    def _flatten(lst):
        if len(lst) > 0 and isinstance(lst[0], list):
            lst = [x for sub in lst for x in sub]
        return lst

    out = _flatten(out)

    p_mean = torch.cat([o["p_mean"] for o in out], dim=0)
    p_var  = torch.cat([o["p_var"]  for o in out], dim=0)
    y      = torch.cat([o["y"]      for o in out], dim=0)
    y_hat  = (p_mean > threshold).int()

    return {"p_mean": p_mean, "p_var": p_var, "y_hat": y_hat, "y": y}
