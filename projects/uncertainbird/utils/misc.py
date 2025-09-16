import pickle
from pathlib import Path
import numpy as np
from typing import Literal
import torch
from typing import Union, Tuple, Dict, Any

from uncertainbird.utils.plotting import plot_reliability_diagram


def load_dump(
    path: Union[str, Path], print_stats: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load predictions and targets from a pickle dump file created by DumpPredictionsCallback.

    This function loads the pickle file created by the DumpPredictionsCallback and returns
    the predictions, targets, and metadata in a convenient format.

    Args:
        path: Path to the pickle file containing predictions and targets.
        print_stats: If True, prints basic statistics about the loaded data.

    Returns:
        A tuple containing:
            - predictions (torch.Tensor): Model predictions of shape (num_samples, num_classes)
            - targets (torch.Tensor): Ground truth targets of shape (num_samples, num_classes)
            - metadata (Dict[str, Any]): Dictionary containing metadata about the experiment,
              including model info, experiment details, and data statistics.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    Example:
        >>> predictions, targets, metadata = load_dump("test_predictions_20250827_123456.pkl")
        >>> print(f"Loaded {predictions.shape[0]} samples with {predictions.shape[1]} classes")
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    predictions = data["predictions"]
    targets = data["targets"].to(torch.int64)  # ensure targets are integer type
    metadata = data["metadata"]

    if print_stats:
        print(f"Loaded data with {metadata['total_samples']} samples")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Model info: {metadata['model_info']}")
        print(f"Save logits: {metadata['save_logits']}")

    return predictions, targets, metadata


def prediction_statistics(
    predictions: torch.Tensor, print_stats: bool = False
) -> Dict[str, float]:
    """Compute basic statistics of the predictions tensor.

    Calculates mean, standard deviation, minimum, and maximum values of the predictions.
    Useful for understanding the distribution and range of model predictions.

    Args:
        predictions: The predictions tensor of any shape. Statistics are computed over all elements.
        print_stats: If True, prints the computed statistics to stdout.

    Returns:
        Dictionary containing the following statistics:
            - mean (float): Mean value of all predictions
            - std (float): Standard deviation of all predictions
            - min (float): Minimum prediction value
            - max (float): Maximum prediction value

    Example:
        >>> stats = prediction_statistics(predictions, print_stats=True)
        >>> print(f"Prediction range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    """
    if print_stats:
        print(
            f"Predictions - mean: {predictions.mean().item():.4f}, std: {predictions.std().item():.4f}, min: {predictions.min().item():.4f}, max: {predictions.max().item():.4f}"
        )
    return {
        "mean": predictions.mean().item(),
        "std": predictions.std().item(),
        "min": predictions.min().item(),
        "max": predictions.max().item(),
    }


def print_classwise_statistics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    """Print class-wise statistics of predictions for multilabel classification.

    For each class with at least one positive target, computes and prints statistics
    including mean, standard deviation, min, max predictions, and number of positive samples.
    This is particularly useful for analyzing model behavior on imbalanced datasets.

    Args:
        predictions: The predictions tensor of shape (num_samples, num_classes).
        targets: The targets tensor of shape (num_samples, num_classes) with binary labels.

    Returns:
        None: This function only prints statistics and doesn't return values.

    Note:
        - Only classes with at least one positive target are analyzed
        - Statistics are computed over all samples for each class
        - Useful for identifying class-specific prediction patterns

    Example:
        >>> print_classwise_statistics(predictions, targets)
        Class 0: mean: 0.1234, std: 0.0567, min: 0.0001, max: 0.9876, positive samples: 42
        Class 1: mean: 0.5678, std: 0.1234, min: 0.0123, max: 0.9999, positive samples: 15
    """

    num_classes = predictions.shape[1]
    for class_idx in range(num_classes):
        class_preds = predictions[:, class_idx]
        class_targets = targets[:, class_idx]
        if (
            class_targets.sum() > 0
        ):  # Only consider classes with at least one positive target
            mean_pred = class_preds.mean().item()
            std_pred = class_preds.std().item()
            min_pred = class_preds.min().item()
            max_pred = class_preds.max().item()
            print(
                f"Class {class_idx}: mean: {mean_pred:.4f}, std: {std_pred:.4f}, min: {min_pred:.4f}, max: {max_pred:.4f}, positive samples: {int(class_targets.sum().item())}"
            )


def extract_top_k(
    predictions,
    targets,
    k=5,
    criterion: Literal[
        "probability", "predicted class", "target class"
    ] = "target class",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract all predictions and targets for top-k classes, which have the highest predicted probabilities."""

    num_labels = predictions.shape[1]
    if criterion == "predicted class":
        # Find top-k classes by highest number of predicted positive samples
        sum_preds_per_class = (predictions >= 0.5).sum(dim=0)  # [num_labels]
        _, top_k_class_indices = torch.topk(
            sum_preds_per_class, min(k, num_labels)
        )  # [k]
    elif criterion == "target class":
        # Find top-k classes by highest number of positive samples in targets
        sum_targets_per_class = targets.sum(dim=0)  # [num_labels]
        _, top_k_class_indices = torch.topk(
            sum_targets_per_class, min(k, num_labels)
        )  # [k]
    elif criterion == "probability":
        # Find top-k classes by highest single prediction scores across all samples
        max_preds_per_class = predictions.max(dim=0)[0]  # [num_labels]
        _, top_k_class_indices = torch.topk(
            max_preds_per_class, min(k, num_labels)
        )  # [k]
    top_k_preds = predictions[:, top_k_class_indices]  #
    top_k_targets = targets[:, top_k_class_indices]  #
    return top_k_preds, top_k_targets
