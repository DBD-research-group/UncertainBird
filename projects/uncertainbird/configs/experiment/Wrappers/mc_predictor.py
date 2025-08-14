# mc_predictor.py
import torch
import torch.nn as nn
import pytorch_lightning as pl

def _enable_dropout_only(m: nn.Module):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.train()

class MCDropoutPredictor(pl.LightningModule):
    def __init__(self, base_model: pl.LightningModule, T: int = 10, threshold: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.T = int(T)
        self.threshold = float(threshold)

    # Lightning will move this module to the device; base_model goes with it.

    def on_predict_start(self):
        # Keep BatchNorm etc. frozen, but turn on *only* dropout layers
        self.base_model.eval()
        self.base_model.apply(_enable_dropout_only)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Adjust unpacking to your batch structure if needed
        x, y = batch

        probs_T = []
        with torch.no_grad():
            for _ in range(self.T):
                logits = self.base_model(x)              # [B, 21]
                probs_T.append(torch.sigmoid(logits))    # [B, 21]

        probs_T = torch.stack(probs_T, dim=0)           # [T, B, 21]
        p_mean  = probs_T.mean(dim=0)                   # [B, 21]
        p_var   = probs_T.var(dim=0, unbiased=True)     # [B, 21]
        y_hat   = (p_mean > self.threshold).int()       # [B, 21]

        return {"p_mean": p_mean, "p_var": p_var, "y_hat": y, "y": y}
