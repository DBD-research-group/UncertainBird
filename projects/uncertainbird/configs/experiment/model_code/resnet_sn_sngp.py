from __future__ import annotations
from typing import Optional
import math
import copy
import torch
import torch.nn as nn

# import your SN-ResNet backbone
import models.resnetsn as resnetsn

# ---------------------- Mean-field adjustment ----------------------

def mean_field_logits(
    logits: torch.Tensor,
    covmat: Optional[torch.Tensor] = None,
    mean_field_factor: float = 0.1,
    likelihood: str = "logistic",  # "logistic" (multiclass softmax) or "binary_logistic" (multilabel sigmoid)
    ) -> torch.Tensor:
    """
    Apply mean-field calibration from SNGP to logits using predictive covariance.
    """
    if mean_field_factor < 0:
        return logits
    if covmat is None:
        variances = 1.0
    else:
        variances = torch.diagonal(covmat)  # [B]
    if likelihood == "poisson":
        scale = torch.exp(-variances * mean_field_factor / 2.0)
    else:
        scale = torch.sqrt(1.0 + variances * mean_field_factor)

    if logits.ndim > 1:
        scale = scale.unsqueeze(-1)
    return logits / scale

# ---------------------- SNGP head (buffers for DDP-friendliness) ----------------------

def _random_feature_linear(i_dim: int, o_dim: int, bias: bool = True, require_grad: bool = False) -> nn.Linear:
    m = nn.Linear(i_dim, o_dim, bias=bias)
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2.0 * math.pi)
        m.bias.requires_grad = require_grad
    return m

class SNGPHead(nn.Module):
    """
    Random Features GP head with moving-average precision (Laplace approx).
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        gp_input_dim: int = 128,
        num_inducing: int = 1024,
        num_classes: int = 21,
        gp_kernel_scale: float = 1.0,
        gp_output_bias: float = 0.0,
        layer_norm_eps: float = 1e-12,
        scale_random_features: bool = True,
        normalize_input: bool = True,
        gp_cov_momentum: float = 0.999,
        gp_cov_ridge_penalty: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__()
        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self.gp_input_scale = 1.0 / math.sqrt(gp_kernel_scale)

        # SN on the linear before RF mapping is standard in SNGP
        self.gp_in = nn.utils.spectral_norm(nn.Linear(hidden_size, gp_input_dim, bias=False))
        self.gp_in.weight.requires_grad = False  # fixed random projection

        self.gp_in_norm = nn.LayerNorm(gp_input_dim, eps=layer_norm_eps)
        self.random_feature = _random_feature_linear(gp_input_dim, num_inducing)  # trainable=False by default
        self.gp_out = nn.Linear(num_inducing, num_classes, bias=False)

        # Bias is typically fixed in SNGP implementations
        self.register_buffer("_gp_output_bias", torch.tensor([gp_output_bias] * num_classes, dtype=torch.float32))

        # Precision matrix as a BUFFER so it saves/loads nicely and works in DDP
        init_prec = gp_cov_ridge_penalty * torch.eye(num_inducing, dtype=torch.float32)
        self.register_buffer("precision_matrix", init_prec.clone())  # [M,M]
        self.register_buffer("_init_precision", init_prec.clone())

    @torch.no_grad()
    def reset_cov(self):
        self.precision_matrix.copy_(self._init_precision)

    def _gp_features(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, hidden_size] -> RF features [B, M]
        """
        z = self.gp_in(h)
        if self.normalize_input:
            z = self.gp_in_norm(z)
        phi = self.random_feature(z)           # [B, M]
        phi = torch.cos(phi)
        if self.scale_random_features:
            phi = phi * self.gp_input_scale
        return phi

    def _update_precision(self, phi: torch.Tensor):
        # phi: [B, M]
        # minibatch precision: phi^T phi -> [M, M]
        B = phi.shape[0]
        minibatch = phi.transpose(0, 1) @ phi  # [M, M]
        if self.gp_cov_momentum > 0:
            minibatch = minibatch / float(B)
            self.precision_matrix.mul_(self.gp_cov_momentum).add_(
                (1.0 - self.gp_cov_momentum) * minibatch
            )
        else:
            self.precision_matrix.add_(minibatch)

    def _predictive_cov(self, phi: torch.Tensor) -> torch.Tensor:
        # (precision)^-1 -> feature covariance
        cov_w = torch.linalg.inv(self.precision_matrix)          # [M, M]
        tmp = cov_w @ phi.transpose(0, 1)                        # [M, B]
        gp_cov = (phi @ tmp) * self.gp_cov_ridge_penalty         # [B, B]
        return gp_cov

    def forward(
        self,
        h: torch.Tensor,              # [B, hidden_size]
        return_gp_cov: bool = False,
        update_cov: Optional[bool] = None,
    ):
        if update_cov is None:
            update_cov = self.training

        phi = self._gp_features(h)                    # [B, M]
        if update_cov:
            self._update_precision(phi)

        logits = self.gp_out(phi) + self._gp_output_bias  # [B, C]

        if return_gp_cov:
            cov = self._predictive_cov(phi)           # [B, B]
            return logits, cov
        return logits

# ---------------------- Full model: SN-ResNet + SNGP ----------------------

class ResNetSN_SNGP(nn.Module):
    """
    Wrap SN-ResNet backbone with SNGP head.
    Returns logits. Use BCEWithLogitsLoss (multilabel) or CrossEntropyLoss (multiclass).
    """
    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 21,
        num_channels: int = 3,
        pretrained: bool = False,
        # SNGP params
        hidden_size: int = 2048,
        gp_input_dim: int = 128,
        num_inducing: int = 1024,
        gp_kernel_scale: float = 1.0,
        gp_cov_momentum: float = 0.999,
        gp_cov_ridge_penalty: float = 1e-3,
        mean_field_at_eval: bool = True,
        mean_field_factor: float = 0.1,
        likelihood: str = "binary_logistic",  # "binary_logistic" for multilabel, "logistic" for multiclass
    ):
        super().__init__()
        # 1) backbone
        ctor = {
            "resnet18":  lambda: resnetsn._resnet("resnet18",  resnetsn.BasicBlock, [2,2,2,2],  pretrained, True, num_classes),
            "resnet34":  lambda: resnetsn._resnet("resnet34",  resnetsn.BasicBlock, [3,4,6,3],  pretrained, True, num_classes),
            "resnet50":  lambda: resnetsn._resnet("resnet50",  resnetsn.Bottleneck, [3,4,6,3],  pretrained, True, num_classes),
            "resnet101": lambda: resnetsn._resnet("resnet101", resnetsn.Bottleneck, [3,4,23,3], pretrained, True, num_classes),
            "resnet152": lambda: resnetsn._resnet("resnet152", resnetsn.Bottleneck, [3,8,36,3], pretrained, True, num_classes),
        }[arch]
        backbone = ctor()  # your SN-ResNet that returns features (fc disabled)

        # Optional: adapt first conv to 1-channel inputs if needed
        if num_channels != 3:
            c1 = backbone.conv1
            new = nn.Conv2d(num_channels, c1.out_channels,
                            kernel_size=c1.kernel_size, stride=c1.stride,
                            padding=c1.padding, bias=False)
            with torch.no_grad():
                if c1.weight.shape[1] == 3:
                    w = c1.weight.mean(dim=1, keepdim=True).repeat(1, num_channels, 1, 1)
                    new.weight.copy_(w)
                else:
                    nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = nn.utils.spectral_norm(new)

        self.backbone = backbone
        self.avgpool = backbone.avgpool    # already defined
        # backbone forward already does GAP+flatten; we keep it for clarity

        # 2) SNGP head
        self.sngp = SNGPHead(
            hidden_size=hidden_size,
            gp_input_dim=gp_input_dim,
            num_inducing=num_inducing,
            num_classes=num_classes,
            gp_kernel_scale=gp_kernel_scale,
            gp_cov_momentum=gp_cov_momentum,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        )

        self.mean_field_at_eval = bool(mean_field_at_eval)
        self.mean_field_factor = float(mean_field_factor)
        self.likelihood = likelihood

    @torch.no_grad()
    def reset_cov(self):
        self.sngp.reset_cov()

    def forward(self, x: torch.Tensor, return_gp_cov: bool = False):
        # Your resnetsn returns features after avgpool+flatten
        h = self.backbone(x)  # [B, 2048]
        logits, cov = None, None
        if return_gp_cov or (not self.training and self.mean_field_at_eval):
            logits, cov = self.sngp(h, return_gp_cov=True, update_cov=self.training)
        else:
            logits = self.sngp(h, return_gp_cov=False, update_cov=self.training)

        # Apply mean-field only at eval if requested
        if (not self.training) and self.mean_field_at_eval:
            logits = mean_field_logits(
                logits, cov, mean_field_factor=self.mean_field_factor, likelihood=self.likelihood
            )

        if return_gp_cov:
            return logits, cov
        return logits
