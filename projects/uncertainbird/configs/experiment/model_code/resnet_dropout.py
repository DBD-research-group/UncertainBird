from typing import List, Literal, Tuple, Sequence, Union, Optional
import torch
import torch.nn as nn
import torchvision

ResNetVersion = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

class ResNetClassifier(nn.Module):
    """
    ResNet classifier with block-wise spatial dropout (+ optional head dropout).
    """

    def __init__(
        self,
        baseline_architecture: ResNetVersion,
        num_classes: int,
        num_channels: int = 1,
        pretrained: bool = False,
        # NEW:
        p_block: Union[float, Sequence[float]] = 0.0,  # scalar or 4-tuple for layer1..4
        p_fc: float = 0.0,    # dropout before final Linear
        **kwargs                         
    ):
        super().__init__()
        self.baseline_architecture = baseline_architecture
        self.num_classes = num_classes
        self.num_channels = num_channels

        # pick constructor + weights
        resnet_ctor = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
        }[baseline_architecture]

        weights = None
        if pretrained:
            weight_enum = {
                "resnet18":  torchvision.models.ResNet18_Weights.DEFAULT,
                "resnet34":  torchvision.models.ResNet34_Weights.DEFAULT,
                "resnet50":  torchvision.models.ResNet50_Weights.DEFAULT,
                "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
                "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
            }[baseline_architecture]
            weights = weight_enum

        m = resnet_ctor(weights=weights)  # torchvision ResNet

        # --- adapt first conv if num_channels != 3 (keeps pretrained useful) ---
        if num_channels != m.conv1.in_channels:
            with torch.no_grad():
                old_w = m.conv1.weight       # [out_c, 3, k, k] if pretrained 3ch
                ksize = m.conv1.kernel_size
                stride = m.conv1.stride
                padding = m.conv1.padding
                bias = False

                new_conv = nn.Conv2d(num_channels, old_w.shape[0],
                                     kernel_size=ksize, stride=stride,
                                     padding=padding, bias=bias)

                if pretrained and old_w.shape[1] == 3:
                    # average RGB, then tile to desired channels
                    mean_w = old_w.mean(dim=1, keepdim=True)                    # [out_c,1,k,k]
                    new_w = mean_w.repeat(1, num_channels, 1, 1)               # [out_c,C,k,k]
                    new_conv.weight.copy_(new_w)
                else:
                    # fallback: kaiming init is fine
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

                m.conv1 = new_conv
                # keep bn1 as-is; running stats remain valid enough after small conv change

        # --- replace classifier head (add dropout -> linear) ---
        in_features = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(p_fc), nn.Linear(in_features, num_classes))

        # --- inject spatial dropout AFTER every residual block in layer1..4 ---
        if isinstance(p_block, (list, tuple)):
            assert len(p_block) == 4, "p_block must have 4 values for layer1..4"
            ps = [float(p) for p in p_block]
        else:
            ps = [float(p_block)] * 4

        for p, lname in zip(ps, ["layer1", "layer2", "layer3", "layer4"]):
            seq = getattr(m, lname)
            for i, block in enumerate(seq):
                seq[i] = nn.Sequential(block, nn.Dropout2d(p))  # insert even if p==0.0

        self.model = m

    def forward(
        self,
        x: torch.Tensor = None,
        inputs: torch.Tensor = None,
        input_values: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        image: torch.Tensor = None,
        **kwargs,
    ):
        # pick the first arg that is not None
        if x is None:
            if inputs is not None:
                x = inputs
            elif input_values is not None:
                x = input_values
            elif pixel_values is not None:
                x = pixel_values
            elif image is not None:
                x = image

        if x is None:
            got = list(kwargs.keys())
            raise ValueError(
                "No input tensor provided. Expected one of "
                "[x, inputs, input_values, pixel_values, image]. "
                f"Extra keys: {got}"
            )

        return self.model(x)