import logging
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
                "ECE_Marginal": MultilabelECEMarginal(
                    num_labels=num_labels, n_bins=10
                ),  # New Metric Added
                "ECE@5": MultilabelECETopK(
                    num_labels=num_labels, k=5, n_bins=10
                ),  # Top-k calibration for practical multilabel prediction
                "ECE@10": MultilabelECETopK(num_labels=num_labels, k=10, n_bins=10),
                "ECE@num_labels": MultilabelECETopK(
                    num_labels=num_labels, k=num_labels, n_bins=10
                ),
                "ACE_OvA": MultilabelACE(
                    num_labels=num_labels, n_bins=10
                ),  # Adaptive Calibration Error
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
            edges = torch.linspace(
                0.0,
                1.0,
                self.n_bins + 1,
                dtype=label_preds.dtype,
                device=label_preds.device,
            )
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

            # Bin the predictions - ensure edges are on same device as input
            edges = torch.linspace(
                0.0, 1.0, self.n_bins + 1, dtype=p_j.dtype, device=p_j.device
            )
            bin_idx = torch.bucketize(p_j, edges, right=False) - 1
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)

            # Get state tensors for this label
            bin_sum_p = getattr(self, f"bin_sum_p_{j}")
            bin_count = getattr(self, f"bin_count_{j}")
            bin_correct = getattr(self, f"bin_correct_{j}")

            # Accumulate statistics - ensure same device and dtype as state tensors
            bin_sum_p.index_add_(
                0,
                bin_idx.to(bin_sum_p.device),
                p_j.to(device=bin_sum_p.device, dtype=bin_sum_p.dtype),
            )
            bin_count.index_add_(
                0,
                bin_idx.to(bin_count.device),
                torch.ones_like(bin_idx, dtype=bin_count.dtype).to(bin_count.device),
            )
            bin_correct.index_add_(
                0,
                bin_idx.to(bin_correct.device),
                y_j.long().to(device=bin_correct.device, dtype=bin_correct.dtype),
            )

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
                # Ensure total_ece is on the same device as ece_j
                if valid_labels == 0:
                    total_ece = torch.tensor(
                        0.0, device=ece_j.device, dtype=ece_j.dtype
                    )
                total_ece += ece_j
                valid_labels += 1

        # Return average across valid labels, or 0 if no valid labels
        if valid_labels > 0:
            return total_ece / valid_labels
        else:
            return torch.tensor(0.0)


class MultilabelECETopK(Metric):
    """
    Top-k Multilabel Expected Calibration Error (ECE@k).

    Implements calibration error focusing only on the top-k most frequently predicted classes
    across the entire dataset. This is more relevant for practical multilabel classification
    where we want to assess calibration for the most relevant/active labels.

    The metric:
    1. Identifies the top-k classes with highest average prediction scores across all samples
    2. Computes calibration error using all samples but only for these top-k classes

    This addresses the practical scenario where we care most about calibration for the classes
    that the model predicts most frequently or with highest confidence.

    Args:
        num_labels: Number of labels in the multilabel classification task
        k: Number of top scoring classes to consider for calibration
        n_bins: Number of bins to use when computing the metric (default: 10)
        norm: Norm used to compare empirical and expected probability bins ('l1', 'l2', or 'max')
        **kwargs: Additional keyword arguments for the Metric base class
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        k: int,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.k = k
        self.n_bins = n_bins
        self.norm = norm

        # Store confidences and accuracies for top-k predictions
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

        # Store all predictions and targets for later processing
        # We'll determine top-k classes globally across all samples
        self.confidences.append(preds)
        self.accuracies.append(target)

    def compute(self) -> Tensor:
        """Compute the top-k multilabel calibration error."""
        if not self.confidences:
            return torch.tensor(0.0)

        try:
            # Concatenate all predictions and targets
            all_preds = dim_zero_cat(self.confidences)  # [N_total, num_labels]
            all_targets = dim_zero_cat(self.accuracies)  # [N_total, num_labels]

            # Find top-k classes by highest single prediction scores across all samples
            max_preds_per_class = all_preds.max(dim=0)[0]  # [num_labels]
            _, top_k_class_indices = torch.topk(
                max_preds_per_class, min(self.k, self.num_labels)
            )

            # Process each selected label independently (same binning approach as ECE_OvA)
            all_confidences = []
            all_accuracies = []

            for class_idx in top_k_class_indices:
                label_preds = all_preds[:, class_idx]  # [N_total]
                label_targets = all_targets[:, class_idx]  # [N_total]

                # Skip if no valid samples for this label
                if label_preds.numel() == 0:
                    continue

                # Manual binning (same approach as ECE_OvA)
                edges = torch.linspace(
                    0.0,
                    1.0,
                    self.n_bins + 1,
                    dtype=label_preds.dtype,
                    device=label_preds.device,
                )
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

            if not all_confidences:
                return torch.tensor(0.0)

            confidences = torch.stack(all_confidences)
            accuracies = torch.stack(all_accuracies)

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
            logging.exception("Error computing MultilabelECETopK", Exception)
            return torch.tensor(0.0)


class MultilabelACE(Metric):
    """
    Multilabel Adaptive Calibration Error (ACE).

    Implements adaptive calibration error for multilabel classification using quantile-based
    binning where each bin contains an equal number of samples. This is different from
    uniform binning (ECE) where bins have equal width.

    The metric uses quantile-based binning strategy:
    - Bin boundaries are determined by data distribution
    - Each bin contains approximately the same number of predictions
    - More adaptive to actual prediction distribution

    As described in calibration literature, ACE can be more robust than ECE when
    predictions are not uniformly distributed across the [0,1] interval.

    Args:
        num_labels: Number of labels in the multilabel classification task
        n_bins: Number of bins to use when computing ACE (default: 10)
        norm: Norm used to compare empirical and expected probability bins ('l1', 'l2', or 'max')
        **kwargs: Additional keyword arguments for the Metric base class
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.n_bins = n_bins
        self.norm = norm

        # Store predictions and targets for quantile-based binning
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

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

        # Store for later computation
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Tensor:
        """Compute the adaptive calibration error using quantile-based binning."""
        if not self.preds:
            return torch.tensor(0.0)

        try:
            # Concatenate all predictions and targets
            all_preds = dim_zero_cat(self.preds)  # [N_total, num_labels]
            all_targets = dim_zero_cat(self.targets)  # [N_total, num_labels]

            # Compute ACE for each label independently (OvA decomposition)
            ace_values = []

            for label_idx in range(self.num_labels):
                label_preds = all_preds[:, label_idx]  # [N_total]
                label_targets = all_targets[:, label_idx]  # [N_total]

                # Skip if no valid samples for this label
                if label_preds.numel() == 0:
                    continue

                # Skip if no variation in targets
                if len(torch.unique(label_targets)) < 2:
                    continue

                # Quantile-based binning: each bin contains equal number of samples
                n_samples = label_preds.shape[0]
                samples_per_bin = n_samples // self.n_bins

                # Sort predictions and targets by prediction values
                sorted_indices = torch.argsort(label_preds)
                sorted_preds = label_preds[sorted_indices]
                sorted_targets = label_targets[sorted_indices]

                bin_confidences = []
                bin_accuracies = []
                bin_weights = []

                for bin_idx in range(self.n_bins):
                    # Define bin boundaries based on sample quantiles
                    start_idx = bin_idx * samples_per_bin
                    if bin_idx == self.n_bins - 1:
                        # Last bin gets remaining samples
                        end_idx = n_samples
                    else:
                        end_idx = (bin_idx + 1) * samples_per_bin

                    if start_idx >= end_idx:
                        continue

                    bin_preds = sorted_preds[start_idx:end_idx]
                    bin_targets = sorted_targets[start_idx:end_idx]

                    if len(bin_preds) == 0:
                        continue

                    # Average confidence in this bin
                    avg_confidence = bin_preds.mean()

                    # Accuracy in this bin
                    accuracy = bin_targets.float().mean()

                    # Weight by number of samples in bin
                    weight = len(bin_preds) / n_samples

                    bin_confidences.append(avg_confidence)
                    bin_accuracies.append(accuracy)
                    bin_weights.append(weight)

                if not bin_confidences:
                    continue

                # Convert to tensors
                confidences = torch.stack(bin_confidences)
                accuracies = torch.stack(bin_accuracies)
                weights = torch.tensor(
                    bin_weights, dtype=confidences.dtype, device=confidences.device
                )

                # Check for NaN values
                if (
                    not torch.isfinite(confidences).all()
                    or not torch.isfinite(accuracies).all()
                ):
                    continue

                # Compute weighted calibration error for this label
                if self.norm == "l1":
                    ace_label = (weights * torch.abs(confidences - accuracies)).sum()
                elif self.norm == "l2":
                    ace_label = (weights * torch.pow(confidences - accuracies, 2)).sum()
                elif self.norm == "max":
                    ace_label = torch.abs(confidences - accuracies).max()
                else:
                    ace_label = (weights * torch.abs(confidences - accuracies)).sum()

                if torch.isfinite(ace_label):
                    ace_values.append(ace_label)

            # Return average ACE across valid labels
            if ace_values:
                result = torch.stack(ace_values).mean()
                return result
            else:
                return torch.tensor(0.0)

        except Exception:
            # Fallback to 0 if computation fails
            return torch.tensor(0.0)
