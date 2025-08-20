# projects/uncertainbird/configs/experiment/model_code/factories.py
import math
from typing import Optional, Dict, Callable
import torch
import torch.nn as nn

from projects.uncertainbird.configs.experiment.model_code import resnetsn as resnetsn_mod
from projects.uncertainbird.configs.experiment.model_code import sngp as sngp_mod

_BACKBONES: Dict[str, Callable[..., nn.Module]] = {
    "resnet50": resnetsn_mod.resnet50,
}

def _make_backbone(name: str, *, num_classes: int, pretrained: bool, progress: bool) -> nn.Module:
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {list(_BACKBONES.keys())}")
    return _BACKBONES[name](pretrained=pretrained, progress=progress, num_classes=num_classes)

class ResNetSNGP(nn.Module):
    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 2,
        num_channels: int = 3,
        pretrained: bool = False,
        progress: bool = True,
        # SNGP head hyperparams
        hidden_size: int = 2048,
        gp_kernel_scale: float = 1.0,
        num_inducing: int = 1024,
        gp_output_bias: float = 0.0,
        layer_norm_eps: float = 1e-12,
        scale_random_features: bool = True,
        normalize_input: bool = True,
        gp_input_dim: int = 128,
        gp_cov_momentum: float = 0.999,
        gp_cov_ridge_penalty: float = 1e-3,
        epochs: int = 40,
        # eval-time flags (aliases)
        mean_field_at_eval: Optional[bool] = None,
        mean_field: Optional[bool] = None,
        mean_field_logits: Optional[bool] = None,
        mean_field_factor: float = math.pi / 8.0,
        num_mc_samples_eval: int = 0,
        # tolerate extra keys (likelihood, pretrain_info, etc.)
        **kwargs,
    ):
        super().__init__()

        # Resolve mean-field alias
        _mf = False
        for flag in (mean_field_at_eval, mean_field, mean_field_logits):
            if isinstance(flag, bool):
                _mf = flag
                break

        # Backbone
        backbone = _make_backbone(arch, num_classes=num_classes, pretrained=pretrained, progress=progress)

        # Adapt first conv for non-3ch inputs (preserves pretrained weights)
        if hasattr(backbone, "conv1") and isinstance(backbone.conv1, nn.Conv2d):
            c_in = backbone.conv1.in_channels
            if num_channels != c_in:
                with torch.no_grad():
                    old = backbone.conv1
                    new = nn.Conv2d(num_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
                    if pretrained and c_in == 3 and old.weight.shape[1] == 3:
                        mean_w = old.weight.mean(dim=1, keepdim=True)
                        new.weight.copy_(mean_w.repeat(1, num_channels, 1, 1))
                    else:
                        nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
                    backbone.conv1 = nn.utils.spectral_norm(new)

        # SNGP head (its internal buffers must be registered there)
        self.model = sngp_mod.SNGP(
            backbone=backbone,
            hidden_size=hidden_size,
            gp_kernel_scale=gp_kernel_scale,
            num_inducing=num_inducing,
            gp_output_bias=gp_output_bias,
            layer_norm_eps=layer_norm_eps,
            scale_random_features=scale_random_features,
            normalize_input=normalize_input,
            gp_input_dim=gp_input_dim,
            gp_cov_momentum=gp_cov_momentum,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            epochs=epochs,
            num_classes=num_classes,
            # no device arg here
            mean_field_at_eval=_mf,
            mean_field_factor=mean_field_factor,
            num_mc_samples_eval=num_mc_samples_eval,
        )

        if kwargs:
            print(f"[ResNetSNGP] Ignoring extra kwargs: {list(kwargs.keys())}")

    def forward(self, *args, **kwargs):
        import torch
        x = None
        # positional first
        for a in args:
            if isinstance(a, torch.Tensor):
                x = a; break
        # common aliases
        if x is None:
            for k in ("x","inputs","input_values","pixel_values","image","spectrogram","waveform"):
                v = kwargs.get(k)
                if isinstance(v, torch.Tensor):
                    x = v; break
        if x is None:
            raise ValueError("No input tensor provided.")

        # optional flags
        return_gp_cov = bool(kwargs.get("return_gp_cov", False))
        update_cov = bool(kwargs.get("update_cov", self.training))

        # channel fix if needed (mono->RGB)
        in_ch = getattr(getattr(self.model.backbone, "conv1", None), "in_channels", None)
        if x.dim() == 4 and in_ch is not None and x.shape[1] != in_ch:
            if x.shape[1] == 1 and in_ch == 3:
                x = x.repeat(1, 3, 1, 1)
            else:
                raise ValueError(f"Expected {in_ch} channels, got {x.shape[1]}.")

        return self.model(x, return_gp_cov=return_gp_cov, update_cov=update_cov)

# (optional) keep function alias for old configs
def build_resnetsn_sngp(**cfg):
    return ResNetSNGP(**cfg)
