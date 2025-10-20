import torch
import torch.nn.functional as F

from uncertainbird.modules.models import UncertainBirdModel

def platt_scaling(
    logits: torch.Tensor, slope: float | torch.Tensor, bias: float | torch.Tensor
) -> torch.Tensor:
    """Apply Platt scaling in logit space (same parameters for all classes)."""
    return logits * slope + bias


def fit_global_platt_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 500,
    lr: float = 0.01,
) -> tuple[float, float]:
    """Optimize a single slope and bias for Platt scaling via BCE loss."""

    device = logits.device
    logits_detached = logits.detach()
    targets_detached = targets.detach().float().to(device)

    parameters = torch.tensor([1.0, 0.0], device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        slope, bias = parameters[0], parameters[1]
        loss = F.binary_cross_entropy_with_logits(
            platt_scaling(logits_detached, slope, bias), targets_detached
        )
        loss.backward()
        return loss

    optimizer.step(_closure)

    slope, bias = parameters.detach().tolist()
    return float(slope), float(bias)

def fit_per_class_platt_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 500,
    lr: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize a slope and bias per class for Platt scaling via BCE loss."""

    device = logits.device
    logits_detached = logits.detach()
    targets_detached = targets.detach().float().to(device)

    slopes = torch.ones(logits.shape[1], device=device, requires_grad=True)
    biases = torch.zeros(logits.shape[1], device=device, requires_grad=True)
    parameters = torch.nn.ParameterList([slopes, biases])
    optimizer = torch.optim.Adam(parameters, lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        slopes, biases = parameters[0], parameters[1]
        loss = F.binary_cross_entropy_with_logits(
            platt_scaling(logits_detached, slopes, biases), targets_detached
        )
        loss.backward()
        optimizer.step()

    return slopes.detach(), biases.detach()


def T_scaling(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    return torch.div(logits, temperature)


def fit_global_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 500,
    lr: float = 0.01,
) -> float:
    """Learn a single temperature by minimizing binary cross-entropy on the calibration set."""

    device = logits.device
    targets_detached = targets.detach().float().to(device)

    temperature = torch.tensor(1.0, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    
    def T_eval() -> float:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(T_scaling(logits, temperature), targets_detached)
        loss.backward()
        return loss

   
    optimizer.step(T_eval)

    return float(temperature.detach().item())


def fit_per_class_temperatures(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 500,
    lr: float = 0.01,
) -> torch.Tensor:
    """Learn a temperature per class by minimizing binary cross-entropy on the calibration set."""
    tars = targets.detach().float()

    log_temperatures = torch.zeros(logits.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([log_temperatures], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperatures = torch.exp(log_temperatures) + 1e-6
        loss = F.binary_cross_entropy_with_logits(logits / temperatures, tars)
        loss.backward()
        optimizer.step()

    return torch.exp(log_temperatures).detach()


def apply_temperature_scaling(
    logits: torch.Tensor, temperature: float | torch.Tensor, model: UncertainBirdModel
) -> torch.Tensor:
    return model.transform_logits_to_probs(logits / temperature)


def apply_platt_scaling(
    logits: torch.Tensor, slope: float | torch.Tensor, bias: float | torch.Tensor, model: UncertainBirdModel
) -> torch.Tensor:
    return model.transform_logits_to_probs(platt_scaling(logits, slope, bias))
