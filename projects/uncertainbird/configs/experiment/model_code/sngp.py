import math
import copy
from statistics import mean
import pdb
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from torch import Tensor


def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    return m


class SNGP(nn.Module):
    def __init__(self, backbone,
                 hidden_size=2048,
                 gp_kernel_scale=1.0,
                 num_inducing=1024,
                 gp_output_bias=0.,
                 layer_norm_eps=1e-12,
                 scale_random_features=True,
                 normalize_input=True,
                 gp_input_dim=128,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 epochs=40,
                 device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"),
                 num_classes=3,
                 **kwargs):
        super(SNGP, self).__init__()
        self.backbone = backbone
        self.final_epochs = epochs - 1
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.pooled_output_dim = hidden_size
        
        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_layer = nn.utils.spectral_norm(nn.Linear(hidden_size, gp_input_dim, bias=False))
        # freeze weight
        self._gp_input_layer.weight.requires_grad = False

        self._gp_input_normalize_layer = torch.nn.LayerNorm(gp_input_dim, eps=layer_norm_eps)
        self._gp_output_layer = nn.Linear(num_inducing, num_classes, bias=False)
        # Register as BUFFER so it moves with .to(device)
        self.register_buffer("_gp_output_bias", torch.full((num_classes,), self.gp_output_bias))
        self._random_feature = RandomFeatureLinear(gp_input_dim, num_inducing)

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        _init = self.gp_cov_ridge_penalty * torch.eye(num_inducing)
         # Register BOTH as BUFFERS (not Parameters)
        self.register_buffer("initial_precision_matrix", _init)
        self.register_buffer("precision_matrix", _init.clone())
    
    def gp_layer(self, gp_inputs: torch.Tensor, update_cov: bool = True):
        """
        Forward through the SNGP head.
        - Normalizes/project inputs
        - Applies random Fourier features + cosine
        - Adds (properly colocated) output bias
        - Optionally updates the precision matrix (train only)

        Returns:
            gp_feature: [B, num_inducing]
            gp_output:  [B, num_classes]
        """
        # Project/normalize inputs
        gp_inputs = self._gp_input_layer(gp_inputs)
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        # Random features + cosine
        gp_feature = self._random_feature(gp_inputs)
        gp_feature = torch.cos(gp_feature)

        # (optional) feature scaling
        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale
            # If you intended to use gp_feature_scale too, you can also:
            # gp_feature = gp_feature * self.gp_feature_scale

        # Linear head + bias (make sure bias is on the same device/dtype)
        out = self._gp_output_layer(gp_feature)
        bias = self._gp_output_bias
        if not isinstance(bias, torch.Tensor) or bias.device != out.device or bias.dtype != out.dtype:
            bias = torch.as_tensor(bias, device=out.device, dtype=out.dtype)
        gp_output = out + bias

        # Update precision only when training (and only if requested)
        if update_cov and self.training:
            with torch.no_grad():
                self.update_cov(gp_feature)

        return gp_feature, gp_output


    @torch.no_grad()
    def reset_cov(self):
        self.precision_matrix.copy_(self.initial_precision_matrix)

    @torch.no_grad()
    def update_cov(self, gp_feature):
        batch_size = gp_feature.size(0)
        P_mb = gp_feature.t() @ gp_feature
        if self.gp_cov_momentum > 0:
            P_mb = P_mb / batch_size
            P_new = self.gp_cov_momentum * self.precision_matrix + (1.0 - self.gp_cov_momentum) * P_mb
        else:
            P_new = self.precision_matrix + P_mb
        self.precision_matrix.copy_(P_new)

    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix

    def forward(self, inputs, return_gp_cov: bool = False,
                update_cov: bool = True):
        # pdb.set_trace()
        x = self.backbone(inputs)
        
        # pdb.set_trace()
        gp_feature, gp_output = self.gp_layer(x, update_cov=update_cov)
        if return_gp_cov:
            gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
            return gp_output, gp_cov_matrix
        # pdb.set_trace()
        return gp_output