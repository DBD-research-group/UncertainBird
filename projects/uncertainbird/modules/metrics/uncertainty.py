import logging
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection, MaxMetric
from torchmetrics.classification import AUROC, BinaryCalibrationError
from torchmetrics.utilities.data import dim_zero_cat
from uncertainbird.utils.misc import extract_top_k


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
                "ECE": MultilabelCalibrationError(n_bins=10),
                "ECE@5": TopKMultiLabelCalibrationError(
                    k=5, n_bins=10
                ),  # Top-k calibration for practical multilabel prediction
                "ECE@10": TopKMultiLabelCalibrationError(k=10, n_bins=10),
            }
        )


class MultilabelCalibrationError(BinaryCalibrationError):
    """
    Computes the Expected Calibration Error (ECE) for multilabel classification tasks.

    This metric extends BinaryCalibrationError to the multilabel setting, supporting different averaging strategies:
      - 'marginal': Computes ECE for each class independently and returns the unweighted mean across classes.
      - 'weighted': Computes ECE for each class and returns the mean weighted by the number of positive samples per class.
      - 'global': Flattens all predictions and targets, treating the multilabel problem as a single binary problem.

    Args:
        n_bins (int): Number of bins to use for calibration. Default is 15.
        norm (str): Norm to use for calibration error ('l1', 'l2', or 'max'). Default is 'l1'.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default is None.
        validate_args (bool): If True, validates the input arguments. Default is True.
        multilabel_average (str, optional): Specifies the type of averaging performed on the data. Options are 'marginal', 'weighted', or 'global'. Default is 'marginal'.
        **kwargs: Additional keyword arguments passed to BinaryCalibrationError.

    Example:
        >>> from torch import tensor
        >>> metric = MultilabelCalibrationError(n_bins=10, norm='l1', multilabel_average='marginal')
        >>> preds = tensor([[0.2, 0.8], [0.6, 0.4]])
        >>> targets = tensor([[0, 1], [1, 0]])
        >>> metric.update(preds, targets)
        >>> ece = metric.compute()
        >>> print(ece)
    """

    def __init__(
        self,
        n_bins=15,
        norm="l1",
        ignore_index=None,
        validate_args=True,
        multilabel_average: Optional[
            Literal["marginal", "weighted", "global"]
        ] = "marginal",
        **kwargs
    ):
        super().__init__(n_bins, norm, ignore_index, validate_args, **kwargs)
        self.multilabel_average = multilabel_average

    def update(self, preds, target):
        # check if preds are in [0, 1] if not apply sigmoid
        if preds.is_floating_point():
            if not torch.all((preds >= 0) & (preds <= 1)):
                preds = torch.sigmoid(preds)

        self.confidences.append(preds)
        self.accuracies.append(target)

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return multilabel_calibration_error(
            confidences,
            accuracies,
            n_bins=self.n_bins,
            norm=self.norm,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            multilabel_average=self.multilabel_average,
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


class TopKMultiLabelCalibrationError(MultilabelCalibrationError):
    """
    Computes the Top-K Expected Calibration Error (ECE) for multilabel classification tasks.

    This metric extends MultilabelCalibrationError by restricting the ECE computation to the top-K classes, selected according to a specified criterion. This is useful for evaluating calibration on the most relevant or confident classes, which often matter most in multilabel settings with many possible classes.

    Selection Criteria (``criterion``):
        - 'probability': Selects the K classes with the highest single prediction scores across all samples (default).
        - 'predicted class': Selects the K classes with the highest number of predicted positive samples (predictions >= 0.5).
        - 'target class': Selects the K classes with the highest number of positive ground truth samples.

    Args:
        n_bins (int): Number of bins to use for calibration. Default is 15.
        norm (str): Norm to use for calibration error ('l1', 'l2', or 'max'). Default is 'l1'.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default is None.
        validate_args (bool): If True, validates the input arguments. Default is True.
        multilabel_average (str, optional): Specifies the type of averaging performed on the data. Options are 'marginal', 'weighted', or 'global'. Default is 'marginal'.
        k (int): Number of top classes to consider for ECE computation. Default is 5.
        criterion (str): Criterion for selecting top-K classes ('probability', 'predicted class', or 'target class'). Default is 'target-class'.
        **kwargs: Additional keyword arguments passed to MultilabelCalibrationError.

    Example:
        >>> from torch import tensor
        >>> metric = TopKMultiLabelCalibrationError(n_bins=10, k=3, criterion='probability')
        >>> preds = tensor([[0.2, 0.8, 0.5], [0.6, 0.4, 0.9]])
        >>> targets = tensor([[0, 1, 1], [1, 0, 0]])
        >>> metric.update(preds, targets)
        >>> ece = metric.compute()
        >>> print(ece)

    Note:
        - Only the top-K classes (according to the selected criterion) are used for ECE computation.
        - The averaging strategy (``multilabel_average``) is applied after selecting the top-K classes.
        - This metric is especially useful for large multilabel problems where only a subset of classes are of primary interest.
    """

    def __init__(
        self,
        n_bins: int = 15,
        norm: str = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        multilabel_average: Optional[
            Literal["marginal", "weighted", "global"]
        ] = "marginal",
        k: int = 5,
        criterion: Literal[
            "probability", "predicted class", "target class"
        ] = "target class",
        **kwargs
    ):
        super().__init__(
            n_bins, norm, ignore_index, validate_args, multilabel_average, **kwargs
        )
        self.k = k
        self.criterion = criterion

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        topk_confidences, topk_accuracies = extract_top_k(
            confidences, accuracies, k=self.k, criterion=self.criterion
        )

        return multilabel_calibration_error(
            topk_confidences,
            topk_accuracies,
            n_bins=self.n_bins,
            norm=self.norm,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            multilabel_average=self.multilabel_average,
        )
