import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection, MaxMetric
from torchmetrics.classification import AUROC

from birdset.modules.metrics.multilabel import mAP, cmAP, cmAP5, pcmAP, TopKAccuracy

from typing import Any, Literal

from torchmetrics.functional.classification.calibration_error import (
    _binary_calibration_error_update,
    _ce_compute,
)
from torchmetrics.utilities.data import dim_zero_cat


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
                "ECE_OvA": MultilabelCalibrationError(num_labels=num_labels, n_bins=10),
                # "EVEC_Marginal": MultilabelECEMarginal(num_labels=num_labels, n_bins=10),  # New Metric Added
            }
        )


class MultilabelCalibrationError(Metric):
    r"""Multilabel Calibration Error.

    Implements calibration error for multilabel classification by treating each label independently
    following the One-vs-All (OvA) decomposition approach. Each binary classification task
    should be calibrated individually.

    The metric computes:
    ECE = Σ_j E[|P[Y_j|φ_j(X) = η_j] - η_j|]

    where each label j is treated as an independent binary classification problem.

    Args:
        num_labels: Number of labels in the multilabel classification task
        n_bins: Number of bins to use when computing the metric (default: 15)
        norm: Norm used to compare empirical and expected probability bins ('l1', 'l2', or 'max')
        **kwargs: Additional keyword arguments for the Metric base class
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.n_bins = n_bins
        self.norm = norm

        # Store confidences and accuracies for each label
        self.add_state("confidences", default=[], dist_reduce_fx="cat")
        self.add_state("accuracies", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds: Predictions of shape (N, num_labels) with values in [0,1] or logits
            target: Binary targets of shape (N, num_labels) with values in {0,1}
        """
        # Auto-sigmoid if looks like logits
        if (preds.min() < 0) or (preds.max() > 1):
            preds = torch.sigmoid(preds)

        # Ensure preds are in [0,1] and handle edge cases
        preds = preds.clamp(0.0, 1.0)

        # Ensure target is binary
        target = target.long()

        # Process each label independently (OvA decomposition)
        all_confidences = []
        all_accuracies = []

        for label_idx in range(self.num_labels):
            label_preds = preds[:, label_idx]  # [N]
            label_targets = target[:, label_idx]  # [N]

            # Skip if no valid samples for this label
            if label_preds.numel() == 0:
                continue

            # Manual binning to avoid potential issues with _binary_calibration_error_update
            edges = torch.linspace(0.0, 1.0, self.n_bins + 1, dtype=label_preds.dtype)
            bin_idx = torch.bucketize(label_preds, edges, right=False) - 1
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)

            # Collect confidences and accuracies for each bin
            confidences = []
            accuracies = []

            for bin_i in range(self.n_bins):
                mask = bin_idx == bin_i
                if not mask.any():
                    continue

                bin_preds = label_preds[mask]
                bin_targets = label_targets[mask]

                # Average confidence in this bin
                avg_conf = bin_preds.mean()

                # Accuracy in this bin
                accuracy = bin_targets.float().mean()

                confidences.append(avg_conf)
                accuracies.append(accuracy)

            if confidences:  # Only add if we have valid bins
                all_confidences.extend(confidences)
                all_accuracies.extend(accuracies)

        # Only update if we have valid data
        if all_confidences:
            self.confidences.append(torch.stack(all_confidences))
            self.accuracies.append(torch.stack(all_accuracies))

    def compute(self) -> Tensor:
        """Compute the multilabel calibration error."""
        if not self.confidences:
            return torch.tensor(0.0)

        try:
            confidences = dim_zero_cat(self.confidences)
            accuracies = dim_zero_cat(self.accuracies)

            # Check for NaN values
            if (
                not torch.isfinite(confidences).all()
                or not torch.isfinite(accuracies).all()
            ):
                return torch.tensor(0.0)

            # Compute calibration error using torchmetrics function
            result = _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

            # Ensure result is finite
            if not torch.isfinite(result):
                return torch.tensor(0.0)

            return result

        except Exception:
            # Fallback to 0 if computation fails
            return torch.tensor(0.0)


class MultilabelECEMarginal(Metric):
    """
    Marginal Multilabel Expected Calibration Error.

    Follows the paper's approach: "In accordance with the OvA decomposition, we could consider the
    labels independently. We define the marginal calibration as ∀j: E[Yj|φj(X) = ηj] = ηj"

    This treats each label as an independent binary classification and computes ECE
    for each label separately, then averages the results.
    """

    def __init__(self, num_labels: int, n_bins: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.n_bins = n_bins

        # Create separate states for each label
        for j in range(num_labels):
            self.add_state(
                f"bin_sum_p_{j}", default=torch.zeros(n_bins), dist_reduce_fx="sum"
            )
            self.add_state(
                f"bin_count_{j}",
                default=torch.zeros(n_bins, dtype=torch.long),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"bin_correct_{j}",
                default=torch.zeros(n_bins, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: [N, num_labels] probabilities or logits
            targets: [N, num_labels] binary targets {0,1}
        """
        # Auto-sigmoid if needed
        if (preds.min() < 0) or (preds.max() > 1):
            preds = torch.sigmoid(preds)

        preds = preds.clamp(0.0, 1.0)
        targets = targets.bool()

        # Process each label independently
        for j in range(self.num_labels):
            p_j = preds[:, j]  # [N]
            y_j = targets[:, j]  # [N]

            # Bin the predictions
            edges = torch.linspace(0.0, 1.0, self.n_bins + 1, dtype=p_j.dtype)
            bin_idx = torch.bucketize(p_j, edges, right=False) - 1
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)

            # Get state tensors for this label
            bin_sum_p = getattr(self, f"bin_sum_p_{j}")
            bin_count = getattr(self, f"bin_count_{j}")
            bin_correct = getattr(self, f"bin_correct_{j}")

            # Accumulate statistics
            bin_sum_p.index_add_(0, bin_idx, p_j)
            bin_count.index_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long))
            bin_correct.index_add_(0, bin_idx, y_j.long())

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        """Compute marginal ECE averaged across all labels."""
        total_ece = torch.tensor(0.0)
        valid_labels = 0

        for j in range(self.num_labels):
            bin_sum_p = getattr(self, f"bin_sum_p_{j}")
            bin_count = getattr(self, f"bin_count_{j}")
            bin_correct = getattr(self, f"bin_correct_{j}")

            # Skip if no data for this label
            if bin_count.sum() == 0:
                continue

            # Compute ECE for label j
            non_empty = bin_count > 0
            if not non_empty.any():
                continue

            # Mean confidence per bin
            bin_conf = torch.zeros_like(bin_sum_p)
            bin_conf[non_empty] = bin_sum_p[non_empty] / bin_count[non_empty].float()

            # Accuracy per bin
            bin_acc = torch.zeros_like(bin_correct, dtype=torch.float)
            bin_acc[non_empty] = (
                bin_correct[non_empty].float() / bin_count[non_empty].float()
            )

            # Bin weights
            total_samples = bin_count.sum().float().clamp_min(1.0)
            bin_weight = bin_count.float() / total_samples

            # ECE for this label
            ece_j = (bin_weight * (bin_conf - bin_acc).abs()).sum()

            # Check for NaN and skip if present
            if torch.isfinite(ece_j):
                total_ece += ece_j
                valid_labels += 1

        # Return average across valid labels, or 0 if no valid labels
        if valid_labels > 0:
            return total_ece / valid_labels
        else:
            return torch.tensor(0.0)
