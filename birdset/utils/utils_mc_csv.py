# utils_mc_csv.py
from typing import List, Optional
import torch
import pandas as pd

def save_mc_predictions_csv(
    p_mean: torch.Tensor,                   # [N, C] probs (after MC average)
    y: Optional[torch.Tensor] = None,       # [N, C] binary targets or None
    p_var: Optional[torch.Tensor] = None,   # [N, C] MC predictive variance (optional)
    class_names: Optional[List[str]] = None,
    out_path: str = "mc_predictions.csv",
    long_format: bool = False,              # False = wide (one row per sample), True = long (one row per (sample,class))
) -> str:
    """
    Saves MC-averaged probabilities (and optionally variance) with targets to CSV.
    Returns the output path.
    """
    assert p_mean.ndim == 2, f"expected [N, C], got {tuple(p_mean.shape)}"
    N, C = p_mean.shape
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    else:
        assert len(class_names) == C, "len(class_names) must match number of columns"

    # Move to CPU numpy
    probs_np = p_mean.detach().cpu().numpy()
    var_np   = p_var.detach().cpu().numpy() if p_var is not None else None
    y_np     = y.detach().cpu().numpy() if y is not None else None

    if not long_format:
        # ---- Wide format: one row per sample; columns prob_xxx, (var_xxx), target_xxx ----
        cols_prob = [f"prob_{c}" for c in class_names]
        df = pd.DataFrame(probs_np, columns=cols_prob)
        if var_np is not None:
            cols_var = [f"var_{c}" for c in class_names]
            df_var = pd.DataFrame(var_np, columns=cols_var)
            df = pd.concat([df, df_var], axis=1)
        if y_np is not None:
            cols_tgt = [f"target_{c}" for c in class_names]
            df_tgt = pd.DataFrame(y_np.astype(int), columns=cols_tgt)
            df = pd.concat([df, df_tgt], axis=1)
        df.insert(0, "sample_idx", range(N))
        df.to_csv(out_path, index=False)
        return out_path

    else:
        # ---- Long format: one row per (sample, class) ----
        # columns: sample_idx, class, prob, [var], [target]
        records = []
        for i in range(N):
            for j, cname in enumerate(class_names):
                rec = {
                    "sample_idx": i,
                    "class": cname,
                    "prob": float(probs_np[i, j]),
                }
                if var_np is not None:
                    rec["var"] = float(var_np[i, j])
                if y_np is not None:
                    rec["target"] = int(y_np[i, j])
                records.append(rec)
        df = pd.DataFrame.from_records(records)
        df.to_csv(out_path, index=False)
        return out_path
