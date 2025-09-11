import logging
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection, MaxMetric
from torchmetrics.classification import AUROC

from birdset.modules.metrics.multilabel import mAP, cmAP, cmAP5, pcmAP, TopKAccuracy

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

from torchmetrics.functional.classification.calibration_error import (
    _binary_calibration_error_update,
    _ce_compute,
)
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from torchmetrics.functional.classification.confusion_matrix import (
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_format,
)


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
    r"""`Top-label Calibration Error`_ for multilabel tasks.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \text{ECE} = \sum_i^N b_i \|(p_i - c_i)\|, \text{L1 norm (Expected Calibration Error)}

    .. math::
        \text{MCE} =  \max_{i} (p_i - c_i), \text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \text{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}, \text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcce`` (:class:`~torch.Tensor`): A scalar tensor containing the calibration error

    Args:
        num_labels: Integer specifying the number of classes
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (multi-label):
        >>> from torch import tensor
        >>> # Example with 4 samples and 3 labels (multi-label: targets are 0/1 per label)
        >>> preds = tensor([[0.25, 0.20, 0.55],
        ...                 [0.55, 0.05, 0.40],
        ...                 [0.10, 0.30, 0.60],
        ...                 [0.90, 0.05, 0.05]])
        >>> target = tensor([[1, 0, 0],
        ...                 [0, 1, 0],
        ...                 [0, 0, 1],
        ...                 [1, 0, 0]])
        >>> metric = MultilabelCalibrationError(num_labels=3, n_bins=3, norm='l1')
        >>> metric(preds, target)
        tensor(0.2222)
        >>> mlce = MultilabelCalibrationError(num_labels=3, n_bins=3, norm='l2')
        >>> mlce(preds, target)
        tensor(0.1235)
        >>> mlce = MultilabelCalibrationError(num_labels=3, n_bins=3, norm='max')
        >>> mlce(preds, target)
        tensor(0.3333)

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(
        self,
        num_labels: int,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        threshold: Optional[float] = 0.5,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_calibration_error_arg_validation(
                num_labels, n_bins, norm, ignore_index
            )
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")
        self.threshold = threshold

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _multilabel_calibration_error_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )
        preds, target = _multilabel_confusion_matrix_format(
            preds, target, num_labels=self.num_labels, threshold=self.threshold
        )
        confidences, accuracies = _multilabel_calibration_error_update(
            preds, target, self.threshold
        )
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        # compute binary calibration error per label and average
        ce_per_label = []
        for c in range(self.num_labels):
            conf = confidences[:, c]
            acc = accuracies[:, c]
            ce = _ce_compute(conf, acc, self.n_bins, norm=self.norm)
            ce_per_label.append(ce)
        ce_per_label = torch.stack(ce_per_label)
        return ce_per_label.mean()

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import randn, randint
            >>> # Example plotting a single value
            >>> from torchmetrics.classification import MulticlassCalibrationError
            >>> metric = MulticlassCalibrationError(num_labels=3, n_bins=3, norm='l1')
            >>> metric.update(randn(20,3).softmax(dim=-1), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn, randint
            >>> # Example plotting a multiple values
            >>> from torchmetrics.classification import MulticlassCalibrationError
            >>> metric = MulticlassCalibrationError(num_labels=3, n_bins=3, norm='l1')
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randn(20,3).softmax(dim=-1), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


def _multilabel_calibration_error_arg_validation(
    num_labels: int,
    n_bins: int,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: Optional[int] = None,
) -> None:
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(
            f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}"
        )
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(
            f"Expected argument `n_bins` to be an integer larger than 0, but got {n_bins}"
        )
    allowed_norm = ("l1", "l2", "max")
    if norm not in allowed_norm:
        raise ValueError(
            f"Expected argument `norm` to be one of {allowed_norm}, but got {norm}."
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )


def _multilabel_calibration_error_tensor_validation(
    preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int] = None
) -> None:
    _multilabel_confusion_matrix_tensor_validation(
        preds, target, num_labels, ignore_index
    )
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def multilabel_calibration_error(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    threshold: Optional[float] = 0.5,
) -> Tensor:
    r"""Expected Calibration Error (ECE) for multilabel tasks.

    This metric quantifies how well a multilabel model's predicted probabilities match the true label distribution.
    Three different norms are implemented, each corresponding to a variation of the calibration error metric:

    .. math::
                ext{ECE} = \sum_i^N b_i \|(p_i - c_i)\|, \text{L1 norm (Expected Calibration Error)}

    .. math::
                ext{MCE} =  \max_{i} (p_i - c_i), \text{Infinity norm (Maximum Calibration Error)}

    .. math::
                ext{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}, \text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the empirical accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    uniformly in the [0,1] range.

    Accepts the following input tensors (multilabel):

    - ``preds`` (float tensor): ``(N, C, ...)``. Each row contains probabilities or logits for C labels.
      If preds has values outside [0,1], logits are assumed and will be auto-sigmoid/softmaxed.
    - ``target`` (int tensor): ``(N, C, ...)``. Each row contains binary ground truth labels (0 or 1) for C labels.
      Only values in {0,1} are valid (except if `ignore_index` is specified).

    Any additional dimensions ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions, shape (N, C, ...)
        target: Tensor with true binary labels, shape (N, C, ...)
        num_labels: Integer specifying the number of labels/classes
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins ('l1', 'l2', 'max').
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: If True, input arguments and tensors are validated for correctness.

    Example (multilabel):
        >>> from torch import tensor
        >>> preds = tensor([[0.25, 0.20, 0.55],
        ...                 [0.55, 0.05, 0.40],
        ...                 [0.10, 0.30, 0.60],
        ...                 [0.90, 0.05, 0.05]])
        >>> target = tensor([[1, 0, 0],
        ...                 [0, 1, 0],
        ...                 [0, 0, 1],
        ...                 [1, 0, 0]])
        >>> multilabel_calibration_error(preds, target, num_labels=3, n_bins=3, norm='l1')
        tensor(0.2222)
        >>> multilabel_calibration_error(preds, target, num_labels=3, n_bins=3, norm='l2')
        tensor(0.1235)
        >>> multilabel_calibration_error(preds, target, num_labels=3, n_bins=3, norm='max')
        tensor(0.3333)

    """
    if validate_args:
        _multilabel_calibration_error_arg_validation(
            num_labels, n_bins, norm, ignore_index
        )
        _multilabel_calibration_error_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target = _multilabel_confusion_matrix_format(
        preds, target, num_labels=num_labels, threshold=threshold
    )
    # preds, target: (N, C)
    num_labels = preds.shape[1]
    ce_per_label = []
    for c in range(num_labels):
        conf = preds[:, c].float()
        acc = ((preds[:, c] > threshold).float() == target[:, c].float()).float()
        ce = _ce_compute(conf, acc, n_bins, norm)
        ce_per_label.append(ce)
    ce_per_label = torch.stack(ce_per_label)
    return ce_per_label.mean()


def _multilabel_calibration_error_update(
    preds: Tensor,
    target: Tensor,
    threshold: Optional[float] = 0.5,
) -> Tuple[Tensor, Tensor]:
    # Accepts preds: (N, C), target: (N, C) for multilabel
    # If preds are logits, apply sigmoid
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.sigmoid()
    # Compute per-label confidences and accuracies (no flattening)
    # preds, target: (N, C)
    # Confidence: predicted probability per label
    confidences = preds.float()  # (C,)
    # Accuracy: mean correctness per label (thresholded prediction == target)
    accuracies = ((preds > threshold).float() == target.float()).float()
    return confidences.float(), accuracies.float()
