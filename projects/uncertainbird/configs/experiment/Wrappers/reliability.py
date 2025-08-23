# projects/uncertainbird/utils/reliability.py
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

@torch.no_grad()
def multilabel_reliability_details(preds: torch.Tensor, target: torch.Tensor,
                                   bins: int = 10, threshold: float = 0.5):
    if preds.ndim > 2:
        N, C = preds.shape[:2]
        preds  = preds.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)
        target = target.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)

    # treat outside [0,1] as logits
    if (preds.min() < 0) or (preds.max() > 1):
        preds = preds.sigmoid()

    preds  = preds.float().nan_to_num(0.5).clamp_(0, 1)
    target = (target > 0)

    edges = torch.linspace(0., 1., bins + 1, device=preds.device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = torch.bucketize(preds, edges, right=False) - 1
    bin_idx.clamp_(0, bins - 1)

    K, C = bins, preds.shape[1]
    bin_sum_p  = torch.zeros(K, dtype=torch.float32, device=preds.device)
    bin_count  = torch.zeros(K, dtype=torch.float32, device=preds.device)
    bin_tp     = torch.zeros(K, C, dtype=torch.float32, device=preds.device)
    bin_fp     = torch.zeros(K, C, dtype=torch.float32, device=preds.device)

    flat_bins  = bin_idx.reshape(-1)
    flat_probs = preds.reshape(-1)

    bin_sum_p.index_add_(0, flat_bins, flat_probs)
    bin_count.index_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=torch.float32))

    pred_pos = preds >= threshold
    for k in range(K):
        in_k = (bin_idx == k)
        if not in_k.any():
            continue
        pp_k = pred_pos & in_k
        bin_tp[k] += (pp_k & target).sum(dim=0)
        bin_fp[k] += (pp_k & (~target)).sum(dim=0)

    mu_k = torch.zeros(K, dtype=torch.float32, device=preds.device)
    nz   = bin_count > 0
    mu_k[nz] = bin_sum_p[nz] / bin_count[nz]

    denom = bin_tp + bin_fp
    prec = torch.zeros_like(denom)
    mask = denom > 0
    prec[mask] = bin_tp[mask] / denom[mask]
    prec_macro_k = prec.mean(dim=1)

    mass_k = bin_count / bin_count.sum().clamp_min(1.0)
    ece = (mass_k * (prec_macro_k - mu_k).abs()).sum()

    return ece.cpu(), centers.cpu(), mu_k.cpu(), prec_macro_k.cpu(), mass_k.cpu()

@torch.no_grad()
def multilabel_reliability_curves(preds, target, bins=10, threshold=0.5):
    # preds: probs in [0,1]; target: {0,1} with shape [N,C]
    if preds.ndim > 2:
        N, C = preds.shape[:2]
        preds  = preds.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)
        target = target.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)
    preds  = preds.float().nan_to_num(0.5).clamp_(0, 1)
    target = (target > 0)

    edges   = torch.linspace(0., 1., bins+1, device=preds.device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = torch.bucketize(preds, edges, right=False) - 1
    bin_idx.clamp_(0, bins-1)

    K, C = bins, preds.shape[1]
    bin_sum_p   = torch.zeros(K, device=preds.device)
    bin_count   = torch.zeros(K, device=preds.device)
    bin_sum_pos = torch.zeros(K, device=preds.device)  # for empirical positive rate
    bin_tp      = torch.zeros(K, C, device=preds.device)
    bin_fp      = torch.zeros(K, C, device=preds.device)

    flat_bins  = bin_idx.reshape(-1)
    flat_probs = preds.reshape(-1)
    flat_targ  = target.reshape(-1).float()

    bin_sum_p.index_add_(0, flat_bins, flat_probs)
    bin_count.index_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=torch.float32))
    bin_sum_pos.index_add_(0, flat_bins, flat_targ)

    # precision@thr inside each bin
    pred_pos = preds >= threshold
    for k in range(K):
        in_k = (bin_idx == k)
        if not in_k.any():
            continue
        pp_k = pred_pos & in_k
        bin_tp[k] += (pp_k & target).sum(dim=0)
        bin_fp[k] += (pp_k & (~target)).sum(dim=0)

    mu = torch.zeros(K, device=preds.device)
    nz = bin_count > 0
    mu[nz] = bin_sum_p[nz] / bin_count[nz]

    # curve A: empirical positive rate (no threshold)
    pos_rate = torch.zeros(K, device=preds.device)
    pos_rate[nz] = bin_sum_pos[nz] / bin_count[nz]

    # curve B: macro precision@thr
    denom = bin_tp + bin_fp
    prec = torch.zeros_like(denom)
    mask = denom > 0
    prec[mask] = bin_tp[mask] / denom[mask]
    prec_macro = prec.mean(dim=1)

    # masses and ECE variants
    mass = bin_count / bin_count.sum().clamp_min(1.0)
    ece_posrate = (mass * (pos_rate - mu).abs()).sum().item()
    ece_precision = (mass * (prec_macro - mu).abs()).sum().item()
    return centers.cpu(), mu.cpu(), pos_rate.cpu(), prec_macro.cpu(), mass.cpu(), ece_posrate, ece_precision

def save_reliability_plot(preds, target, path_png: str, bins=10, threshold=0.5,
                          title="Reliability (macro, multilabel)") -> dict:
    """
    Save a reliability diagram with both curves:
      - PosRate: empirical positive rate (classic reliability diagram)
      - Precision@thr: macro precision with thresholding
    Returns dict of both ECE values.
    """
    centers, mu, posrate, prec_macro, mass, ece_posrate, ece_precision = multilabel_reliability_curves(
        preds, target, bins=bins, threshold=threshold
    )

    plt.figure(figsize=(5.5, 4.5))
    plt.plot([0,1], [0,1], color="black", linewidth=1)

    sizes = (mass * 1500).clamp_min(10)

    # curve A: empirical positive rate
    plt.scatter(mu, posrate, s=sizes, alpha=0.7, label=f"PosRate (ECE={ece_posrate:.4f})")

    # curve B: macro precision@thr
    plt.scatter(mu, prec_macro, s=sizes, marker="x", c="red", alpha=0.8,
                label=f"Prec@{threshold:.2f} (ECE={ece_precision:.4f})")

    # add bin masses as text
    for x, y, w in zip(mu.tolist(), posrate.tolist(), mass.tolist()):
        plt.text(x + 0.01, y, f"{w:.2f}", fontsize=7)

    plt.xlabel("Mean confidence in bin")
    plt.ylabel("y in bin")
    plt.title(f"{title}\nBins={bins}, thr={threshold}")
    plt.legend()
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

    return {"ece_posrate": float(ece_posrate), "ece_precision": float(ece_precision)}