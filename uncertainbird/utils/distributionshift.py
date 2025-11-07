import pickle
import torch
from pathlib import Path

# Helpers: load latest pickle for (subset, split) and sample embeddings


def latest_pickle_for(subset: str, split: str, output_dir: Path) -> Path | None:
    base = output_dir / subset / split
    if not base.exists():
        return None
    picks = sorted(base.glob("audiomae_embeddings_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return picks[0] if picks else None


def load_clip_embeddings(subset: str, split: str, output_dir: Path) -> torch.Tensor:
    p = latest_pickle_for(subset, split, output_dir)
    if p is None:
        raise FileNotFoundError(f"No pickle found for {subset}/{split} under {output_dir}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    emb = data["clip_embeddings"]  # Tensor [N, D]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    return emb


def sample_embeddings(x: torch.Tensor, max_n: int, seed: int = 42) -> torch.Tensor:
    if x.shape[0] <= max_n:
        return x
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:max_n]
    return x[idx]

def load_embeddings_and_labels(subset: str, split: str, output_dir: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    p = latest_pickle_for(subset, split, output_dir)
    if p is None:
        raise FileNotFoundError(f"No pickle found for {subset}/{split} under {output_dir}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    emb = data["clip_embeddings"]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    labels = data.get("labels", None)
    if labels is not None and not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    return emb, labels




# Metrics: MMD (RBF), Fréchet (FID-style), Energy distance, optional Sinkhorn OT

def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return x2 + y2 - 2.0 * (x @ y.T)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    d2 = pairwise_sq_dists(x, y)
    return torch.exp(-gamma * d2)


def median_heuristic_gamma(x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        # sample a subset for efficiency
        xs = x if x.shape[0] <= 1000 else x[torch.randperm(x.shape[0])[:1000]]
        ys = y if y.shape[0] <= 1000 else y[torch.randperm(y.shape[0])[:1000]]
        d2 = pairwise_sq_dists(xs, ys)
        med = torch.median(d2)
        g = 1.0 / (2.0 * (med.item() + 1e-8))
    return g


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> float:
    gamma = median_heuristic_gamma(x, y)
    Kxx = rbf_kernel(x, x, gamma)
    Kyy = rbf_kernel(y, y, gamma)
    Kxy = rbf_kernel(x, y, gamma)
    n = x.shape[0]
    m = y.shape[0]
    # Unbiased MMD^2
    mmd2 = (Kxx.sum() - Kxx.trace()) / (n * (n - 1) + 1e-8)
    mmd2 += (Kyy.sum() - Kyy.trace()) / (m * (m - 1) + 1e-8)
    mmd2 -= 2.0 * Kxy.mean()
    return float(mmd2.item())


def frechet_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    # Compute mean/cov and Fréchet distance
    mu1 = x.mean(dim=0)
    mu2 = y.mean(dim=0)
    x_c = x - mu1
    y_c = y - mu2
    # cov with unbiased=False for stability
    cov1 = (x_c.T @ x_c) / (x.shape[0] - 1)
    cov2 = (y_c.T @ y_c) / (y.shape[0] - 1)
    diff = mu1 - mu2
    eps = 1e-6
    cov1 = cov1 + eps * torch.eye(cov1.shape[0], device=cov1.device)
    cov2 = cov2 + eps * torch.eye(cov2.shape[0], device=cov2.device)
    # eigenvalues of the product are non-negative in theory; clip numerically
    eigvals = torch.linalg.eigvals(cov1 @ cov2).real.clamp(min=0)
    tr_cov1 = torch.trace(cov1)
    tr_cov2 = torch.trace(cov2)
    tr_sqrt = torch.sum(torch.sqrt(eigvals + eps))
    fid = (diff @ diff).item() + (tr_cov1 + tr_cov2 - 2.0 * tr_sqrt).item()
    return float(fid)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    # E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||
    def mean_pairwise_norm(a, b):
        d2 = pairwise_sq_dists(a, b)
        return torch.mean(torch.sqrt(torch.clamp(d2, min=0)))
    term_xy = mean_pairwise_norm(x, y)
    term_xx = mean_pairwise_norm(x, x)
    term_yy = mean_pairwise_norm(y, y)
    e = 2 * term_xy - term_xx - term_yy
    return float(e.item())


def sinkhorn_ot(x: torch.Tensor, y: torch.Tensor, eps: float = 0.05, iters: int = 200) -> float:
    # Simple entropic OT on uniform measures; returns transport cost
    n, m = x.shape[0], y.shape[0]
    C = pairwise_sq_dists(x, y)  # cost matrix (squared euclidean)
    K = torch.exp(-C / eps)
    u = torch.ones(n, device=x.device) / n
    v = torch.ones(m, device=x.device) / m
    for _ in range(iters):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
    P = torch.diag(u) @ K @ torch.diag(v)
    cost = torch.sum(P * C)
    return float(cost.item())

def compute_metric_per_class(x: torch.Tensor, x_targets: torch.Tensor,
                             y: torch.Tensor, y_targets: torch.Tensor,
                             metric_fn,
                             min_per_class: int,
                             max_per_class: int = 500,
                             seed: int = 42) -> float:
    '''Compute metric_func between x and y per class defined in x_targets and y_targets and than mean over classes.
    If a class is missing in either x_targets or y_targets, the result for that class is NaN.
    Args:
        x: Tensor of shape (num_samples_x, features)
        x_targets: Tensor of shape (num_samples_x, num_classes)
        y: Tensor of shape (num_samples_y, features)
        y_targets: Tensor of shape (num_samples_y, num_classes)
        metric_func: function that takes two Tensors (x_c, y_c) and returns a float
    Returns:
        float: mean metric over classes'''
   
    results = {}
    rng = torch.Generator().manual_seed(seed)
    x_targets = x_targets.int()
    y_targets = y_targets.int()
    # iterate over classes
    for c in range(x_targets.shape[1]):
        # try to select min_per_class samples from x
        x_mask = x_targets[:, c] == 1
        x_c = x[x_mask]
        if x_c.shape[0] < min_per_class:
            continue
        # try to select min_per_class samples from y
        y_mask = y_targets[:, c] == 1
        y_c = y[y_mask]
        if y_c.shape[0] < min_per_class:
            continue
        # sample max_per_class samples from x_c and y_c and min the number of available samples
        n_x = min(x_c.shape[0], max_per_class)
        n_y = min(y_c.shape[0], max_per_class)
        x_c_sampled = x_c[torch.randperm(x_c.shape[0], generator=rng)[:n_x]]
        y_c_sampled = y_c[torch.randperm(y_c.shape[0], generator=rng)[:n_y]]
        # compute metric
        metric = metric_fn(x_c_sampled, y_c_sampled)
        results[c] = metric

    return float(torch.mean(torch.tensor(list(results.values()))))