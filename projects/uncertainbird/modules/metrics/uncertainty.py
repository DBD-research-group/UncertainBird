import logging
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection, MaxMetric
from torchmetrics.classification import AUROC


from birdset.modules.metrics.multilabel import mAP, cmAP, cmAP5, pcmAP, TopKAccuracy

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

from torchmetrics.functional.classification import binary_calibration_error


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


def multilabel_calibration_error(
    preds: Tensor,
    target: Tensor,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    multilabel_average: Optional[
        Literal["marginal", "weighted", "global"]
    ] = "marginal",
) -> Tensor:
    """
    Computes the Expected Calibration Error (ECE) for multilabel classification.

    Args:
        preds (Tensor): Predictions from model (probabilities) with shape (N, C) where C is the number of classes.
        target (Tensor): Ground truth labels with shape (N, C) where each entry is 0 or 1.
        n_bins (int): Number of bins to use for calibration. Default is 15.
        norm (str): Norm to use for calculating the error. Options are 'l1', 'l2', or 'max'. Default is 'l1'.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default is None.
        validate_args (bool): If True, validates the input arguments. Default is True.
        multilabel_average (str, optional): Specifies the type of averaging performed on the data. Options are 'marginal', 'weighted', or 'none'.
        'marginal' computes the ECE for each class and averages them.
        'weighted' computes the ECE for each class and averages them weighted by the number of true instances for each class.
        'global' transforms the multilabel problem into a single binary problem by flattening the predictions and targets.
        Default is 'marginal'.

    Returns:
        Tensor: The computed ECE value.
    """
    if multilabel_average not in ("marginal", "weighted", "global"):
        raise ValueError(
            "multilabel_average must be one of 'marginal', 'weighted', or 'global'"
        )
    if multilabel_average == "global":
        preds = preds.flatten().float()
        target = target.flatten().float()
        ece = binary_calibration_error(
            preds,
            target,
            n_bins=n_bins,
            norm=norm,
            ignore_index=ignore_index,
            validate_args=validate_args,
        )
    elif multilabel_average == "marginal" or multilabel_average == "weighted":
        ece_per_class = []
        positives_per_class = []
        num_classes = preds.shape[1]
        for class_idx in range(num_classes):
            class_preds = preds[:, class_idx].float()
            class_target = target[:, class_idx].float()
            ece_class = binary_calibration_error(
                class_preds,
                class_target,
                n_bins=n_bins,
                norm=norm,
                ignore_index=ignore_index,
                validate_args=validate_args,
            )
            ece_per_class.append(ece_class)
            positives_per_class.append(class_target.sum().item())
        if multilabel_average == "weighted":
            ece = sum(e * p for e, p in zip(ece_per_class, positives_per_class)) / (
                sum(positives_per_class)
            )
        else:  # 'marginal'
            ece = torch.stack(ece_per_class).mean()
    return ece
