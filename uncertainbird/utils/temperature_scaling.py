import torch
import torch.nn.functional as F


def _fit_global_temperature(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 500,
    lr: float = 0.01,
) -> float:
    """Learn a single temperature by minimizing binary cross-entropy on the calibration set."""
    probs = probabilities.detach().float().clamp(1e-6, 1 - 1e-6)
    tars = targets.detach().float()

    logits = torch.logit(probs)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature) + 1e-6
        loss = F.binary_cross_entropy_with_logits(logits / temperature, tars)
        loss.backward()
        optimizer.step()

    return float(torch.exp(log_temperature).detach().item())


def _fit_per_class_temperatures(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 500,
    lr: float = 0.01,
) -> torch.Tensor:
    """Learn a temperature per class by minimizing binary cross-entropy on the calibration set."""
    probs = probabilities.detach().float().clamp(1e-6, 1 - 1e-6)
    tars = targets.detach().float()

    logits = torch.logit(probs)
    log_temperatures = torch.zeros(probabilities.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([log_temperatures], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperatures = torch.exp(log_temperatures) + 1e-6
        loss = F.binary_cross_entropy_with_logits(logits / temperatures, tars)
        loss.backward()
        optimizer.step()

    return torch.exp(log_temperatures).detach()


def _apply_temperature_scaling(
    probabilities: torch.Tensor, temperature: float | torch.Tensor
) -> torch.Tensor:
    probs = probabilities.float().clamp(1e-6, 1 - 1e-6)
    logits = torch.logit(probs)
    return torch.sigmoid(logits / temperature)
