import logging
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection, MaxMetric
from torchmetrics.classification import AUROC
from uncertainbird.utils.plotting import _bin_stats_torch

from birdset.modules.metrics.multilabel import mAP, cmAP, cmAP5, pcmAP, TopKAccuracy

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class MultilabelMetricsConfig:
    """
    A class for configuring the metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(
        self,
        num_labels: int = 21,
    ):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: Metric = cmAP(num_labels=num_labels, thresholds=None)
        self.val_metric_best: Metric = MaxMetric()
        self.add_metrics: MetricCollection = MetricCollection(
            {
                "MultilabelAUROC": AUROC(
                    task="multilabel",
                    num_labels=num_labels,
                    average="macro",
                    thresholds=None,
                ),
                "T1Accuracy": TopKAccuracy(topk=1),
                "T3Accuracy": TopKAccuracy(topk=3),
                "mAP": mAP(num_labels=num_labels, thresholds=None),
            }
        )
        self.eval_complete: MetricCollection = MetricCollection(
            {
                "cmAP5": cmAP5(
                    num_labels=num_labels, sample_threshold=5, thresholds=None
                ),
                "pcmAP": pcmAP(
                    num_labels=num_labels,
                    padding_factor=5,
                    average="macro",
                    thresholds=None,
                ),
                # "ECE_OvA": MultilabelCalibrationError(num_labels=num_labels, n_bins=10),
                # "ECE_Marginal": MultilabelECEMarginal(
                #     num_labels=num_labels, n_bins=10
                # ),  # New Metric Added
                # "ECE@5": MultilabelECETopK(
                #     num_labels=num_labels, k=5, n_bins=10
                # ),  # Top-k calibration for practical multilabel prediction
                # "ECE@10": MultilabelECETopK(num_labels=num_labels, k=10, n_bins=10),
                # "ECE@num_labels": MultilabelECETopK(
                #     num_labels=num_labels, k=num_labels, n_bins=10
                # ),
                # "ACE_OvA": MultilabelACE(
                #     num_labels=num_labels, n_bins=10
                # ),  # Adaptive Calibration Error
            }
        )


class MultilabelCalibrationError(Metric):
    """
    Computes the Expected Calibration Error (ECE) for multilabel classification.

    This metric calculates the ECE by treating all predictions as a single pool (flattened).
    It supports different norms for measuring calibration error: L1, L2, and Max.

    Args:
        num_labels (int): Number of labels in the multilabel classification task.
        n_bins (int): Number of bins to use for calibration. Default is 10.
    """

    def __init__(
        self,
        num_labels: int,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        type: Literal["global", "marginal"] = "global",
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.num_labels = num_labels
        self.n_bins = n_bins
        self.norm = norm
        self.type = type
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update the state with new predictions and targets.

        Args:
            preds (Tensor): Model predictions of shape (N, C) or (N, C, ...).
            targets (Tensor): Ground truth labels of shape (N, C) or (N, C, ...).
        """
        preds = torch.as_tensor(preds, dtype=torch.float32)
        targets = torch.as_tensor(targets, dtype=torch.float32)
        if preds.dim() > 2:
            preds = preds.view(-1, preds.shape[-1])
            targets = targets.view(-1, targets.shape[-1])
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:
        """
        Compute the Expected Calibration Error (ECE).

        Returns:
            torch.Tensor: Scalar ECE value.
        """
        return compute_calibration_error(
            torch.cat(self.preds, dim=0),
            torch.cat(self.targets, dim=0),
            n_bins=self.n_bins,
            norm=self.norm,
            type=self.type,
        )


def compute_calibration_error(
    predictions,
    targets,
    n_bins=10,
    norm: Literal["l1", "l2", "max"] = "l1",
    type: Literal["global", "marginal"] = "global",
) -> torch.Tensor:
    """
    Compute calibration error for multilabel classification.

    Args:
        predictions (Tensor): Model predictions (N, C) or (N, C, ...).
        targets (Tensor): Ground truth labels (N, C) or (N, C, ...).
        n_bins (int): Number of bins to use for calibration. Default is 10.
        norm (str): Norm to use for calibration error ('l1', 'l2', or 'max'). Default is 'l1'.
        type (str): Type of calibration error to compute ('global' or 'marginal').
            'global': Treat all predictions as a single pool (flattened).
            'marginal': Compute ECE per label and average (macro over labels).
    Returns:
        torch.Tensor: Scalar calibration error.
    """
    # Ensure torch tensors and float type
    preds = torch.as_tensor(predictions, dtype=torch.float32)
    targs = torch.as_tensor(targets, dtype=torch.float32)
    if preds.dim() > 2:
        preds = preds.view(-1, preds.shape[-1])
        targs = targs.view(-1, targs.shape[-1])
    N, C = preds.shape

    if type == "global":
        # Flatten all predictions and targets
        conf = preds.flatten()
        labels = targs.flatten()
        bin_confs, bin_accs, bin_weights, ece, mce = _bin_stats_torch(
            conf, labels, n_bins=n_bins, quantile=False
        )
        if norm == "l1":
            return torch.tensor(ece)
        elif norm == "l2":
            # L2: weighted root mean squared calibration gap
            valid = bin_weights > 0
            l2 = (
                (bin_weights[valid] * (bin_accs[valid] - bin_confs[valid]) ** 2)
                .sum()
                .sqrt()
            )
            return l2
        elif norm == "max":
            return torch.tensor(mce)
        else:
            raise ValueError(f"Unknown norm: {norm}")
    elif type == "marginal":
        # Compute ECE per label, then macro-average
        eces = []
        for c in range(C):
            conf = preds[:, c]
            labels = targs[:, c]
            bin_confs, bin_accs, bin_weights, ece, mce = _bin_stats_torch(
                conf, labels, n_bins=n_bins, quantile=False
            )
            if norm == "l1":
                eces.append(ece)
            elif norm == "l2":
                valid = bin_weights > 0
                l2 = (
                    (bin_weights[valid] * (bin_accs[valid] - bin_confs[valid]) ** 2)
                    .sum()
                    .sqrt()
                    .item()
                )
                eces.append(l2)
            elif norm == "max":
                eces.append(mce)
            else:
                raise ValueError(f"Unknown norm: {norm}")
        if norm in ["l1", "l2"]:
            return torch.tensor(eces).mean()
        elif norm == "max":
            return torch.tensor(eces).max()
    else:
        raise ValueError(f"Unknown type: {type}")


def _bin_stats_torch(predictions, targets, n_bins=10, quantile=False):
    """
    Compute bin-wise calibration statistics for reliability diagrams and ECE/MCE using torch.

    This function bins predictions and targets (flattened) into either uniform or quantile bins,
    then computes, for each bin:
      - The mean predicted confidence (average predicted probability)
      - The mean accuracy (fraction of positives)
      - The weight (fraction of total samples in the bin)
      - The calibration gap (absolute difference between accuracy and confidence)
    It then aggregates these to compute:
      - ECE (Expected Calibration Error): weighted average of calibration gaps
      - MCE (Maximum Calibration Error): largest calibration gap across bins

    Args:
        predictions (Tensor or array-like): Model predicted probabilities, shape (N,) or (N, C). Flattened to 1D.
        targets (Tensor or array-like): Ground truth binary labels, same shape as predictions. Flattened to 1D.
        n_bins (int): Number of bins to use for calibration statistics.
        quantile (bool): If True, use quantile bins (equal number of samples per bin). If False, use uniform bins.

    Returns:
        bin_confs (torch.Tensor): Mean predicted confidence per bin, shape (n_bins,)
        bin_accs (torch.Tensor): Mean accuracy per bin, shape (n_bins,)
        bin_weights (torch.Tensor): Fraction of samples in each bin, shape (n_bins,)
        ece (float): Expected Calibration Error (weighted average of bin gaps)
        mce (float): Maximum Calibration Error (largest bin gap)
    """
    # Convert to torch tensors and flatten
    conf = torch.as_tensor(predictions, dtype=torch.float32).reshape(-1)
    labels = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
    n = conf.numel()

    # Choose bin edges: quantile or uniform
    if quantile:
        # Quantile binning: edges chosen so each bin has ~equal number of samples
        qs = torch.linspace(0, 1, n_bins + 1, dtype=conf.dtype, device=conf.device)
        edges = torch.quantile(conf, qs)
        edges[0], edges[-1] = 0.0, 1.0  # Ensure endpoints
        edges = torch.maximum(edges, torch.cummax(edges, 0)[0])  # Ensure non-decreasing
    else:
        # Uniform bins in [0, 1]
        edges = torch.linspace(0, 1, n_bins + 1, dtype=conf.dtype, device=conf.device)

    bin_confs, bin_accs, bin_weights, bin_gaps = [], [], [], []
    for b in range(n_bins):
        left, right = edges[b].item(), edges[b + 1].item()
        # For all but last bin: [left, right)
        # For last bin: [left, right] (include right endpoint)
        if b < n_bins - 1:
            m = (conf >= left) & (conf < right)
        else:
            m = (conf >= left) & (conf <= right)
        if m.any():
            # Compute mean confidence and accuracy for this bin
            c_mean = conf[m].mean().item()
            a_mean = labels[m].mean().item()
            w = m.sum().item() / n  # Fraction of total samples in this bin
            gap = abs(a_mean - c_mean)  # Calibration gap
        else:
            # Empty bin: set confidence to bin midpoint, accuracy NaN, weight/gap 0
            c_mean = (left + right) / 2
            a_mean, w, gap = float("nan"), 0.0, 0.0
        bin_confs.append(c_mean)
        bin_accs.append(a_mean)
        bin_weights.append(w)
        bin_gaps.append(gap)

    # Convert lists to torch tensors
    bin_confs = torch.tensor(bin_confs, dtype=conf.dtype)
    bin_accs = torch.tensor(bin_accs, dtype=conf.dtype)
    bin_weights = torch.tensor(bin_weights, dtype=conf.dtype)
    bin_gaps = torch.tensor(bin_gaps, dtype=conf.dtype)

    # Only consider bins with nonzero weight for ECE/MCE
    valid = bin_weights > 0
    if valid.any():
        ece = float((bin_weights[valid] * bin_gaps[valid]).sum().item())
        mce = float(bin_gaps[valid].max().item())
    else:
        ece = 0.0
        mce = 0.0
    return bin_confs, bin_accs, bin_weights, ece, mce
