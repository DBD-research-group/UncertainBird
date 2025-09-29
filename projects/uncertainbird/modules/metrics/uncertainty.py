import logging
import torch
from torch import Tensor
from typing import Literal, Optional, Union
from torchmetrics import Metric, MetricCollection, MaxMetric
from torchmetrics.classification import AUROC, BinaryCalibrationError
from torchmetrics.functional.classification import binary_calibration_error
from torchmetrics.functional.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _binary_calibration_error_tensor_validation,
    _binary_confusion_matrix_format,
    _binary_calibration_error_update,
    _binning_bucketize,
)
from torchmetrics.utilities.data import dim_zero_cat


from birdset.modules.metrics.multilabel import mAP, cmAP, cmAP5, pcmAP, TopKAccuracy

from uncertainbird.utils.misc import extract_top_k


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


class NoMetricsConfig:
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
        self.add_metrics: MetricCollection = MetricCollection({})
        self.eval_complete: MetricCollection = MetricCollection({})


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
        **kwargs,
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
        **kwargs,
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


def _mcs_compute(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Union[Tensor, int],
    norm: Literal["l1", "l2", "max", "over", "under"] = "l1",
    debias: bool = False,
) -> Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.

    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = torch.linspace(
            0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device
        )

    if norm not in {"l1", "l2", "max", "over", "under"}:
        raise ValueError(
            f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}"
        )

    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(
            confidences, accuracies, bin_boundaries
        )

    if norm == "l1":
        return torch.sum((conf_bin - acc_bin) * prop_bin)
    if norm == "max":
        ce = torch.max(conf_bin - acc_bin)
    if norm == "l2":
        ce = torch.sum(torch.pow(conf_bin - acc_bin, 2) * prop_bin)
        # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
        if debias:
            # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from
            # the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                prop_bin * accuracies.size()[0] - 1
            )
            ce += torch.sum(
                torch.nan_to_num(debias_bins)
            )  # replace nans with zeros if nothing appeared in a bin
        return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    if (
        norm == "under"
    ):  # only sum up negative values leading to quantify under-confidence
        ce = torch.sum(torch.clamp_max((conf_bin - acc_bin), 0) * prop_bin)
    if norm == "over":  # only sum up positive values
        ce = torch.sum(torch.clamp_min((conf_bin - acc_bin), 0) * prop_bin)
    return ce


def binary_miscalibration_score(
    preds: Tensor,
    target: Tensor,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max", "over", "under"] = "l1",
    validate_args: bool = False,
    ignore_index: Optional[int] = None,
) -> Tensor:
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False
    )
    confidences, accuracies = _binary_calibration_error_update(preds, target)
    return _mcs_compute(confidences, accuracies, n_bins, norm)


def multilabel_miscalibration_score(
    preds: Tensor,
    target: Tensor,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max", "over", "under"] = "l1",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    multilabel_average: Optional[
        Literal["marginal", "weighted", "global"]
    ] = "marginal",
) -> Tensor:

    if multilabel_average not in ("marginal", "weighted", "global"):
        raise ValueError(
            "multilabel_average must be one of 'marginal', 'weighted', or 'global'"
        )
    if multilabel_average == "global":
        preds = preds.flatten().float()
        target = target.flatten().float()
        ece = binary_miscalibration_score(
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
            ece_class = binary_miscalibration_score(
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


class MiscalibrationScore(MultilabelCalibrationError):

    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "over", "under"] = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = False,
        multilabel_average: Optional[
            Literal["marginal", "weighted", "global"]
        ] = "marginal",
        **kwargs,
    ):
        super().__init__(
            n_bins,
            norm,
            ignore_index,
            validate_args=False,
            multilabel_average=multilabel_average,
            **kwargs,
        )

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)

        return multilabel_miscalibration_score(
            confidences,
            accuracies,
            n_bins=self.n_bins,
            norm=self.norm,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            multilabel_average=self.multilabel_average,
        )
