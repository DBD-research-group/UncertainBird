import pickle
from pathlib import Path
from typing import Literal
import torch
from typing import Union, Tuple, Dict, Any


from torchmetrics.functional.classification import (
    binary_calibration_error,
    precision,
    recall,
    f1_score,
)


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


def class_wise_statistics(
    predictions: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
) -> Dict[str, torch.Tensor]:
    """Compute class-wise statistics of predictions for multilabel classification.

    For each class with at least one positive target, computes statistics including mean,
    standard deviation, min, max predictions, and number of positive samples.
    This is particularly useful for analyzing model behavior on imbalanced datasets.

    Args:
        predictions: The predictions tensor of shape (num_samples, num_classes).
        targets: The targets tensor of shape (num_samples, num_classes) with binary labels.

    Returns:
        Dictionary containing class-wise statistics:
            - mean (float): Mean prediction value for the class
            - std (float): Standard deviation of prediction values for the class
            - min (float): Minimum prediction value for the class
            - max (float): Maximum prediction value for the class
            - positive_samples (int): Number of positive samples for the class
            - precision (float): Precision for the class
            - recall (float): Recall for the class
            - f1_score (float): F1 score for the class
            - ece (float): Expected Calibration Error for the class
    """

    num_classes = predictions.shape[1]
    stats = {
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
        "positive_samples": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "ece": [],
    }
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
            positive_samples = int(class_targets.sum().item())

            pre = precision(class_preds, class_targets, task="binary").item()
            rec = recall(class_preds, class_targets, task="binary").item()
            f1 = f1_score(class_preds, class_targets, task="binary").item()
            # Compute ECE using reliability diagram
            ece = binary_calibration_error(
                class_preds, class_targets, n_bins=n_bins
            ).item()
            stats["mean"].append(mean_pred)
            stats["std"].append(std_pred)
            stats["min"].append(min_pred)
            stats["max"].append(max_pred)
            stats["positive_samples"].append(positive_samples)
            stats["precision"].append(pre)
            stats["recall"].append(rec)
            stats["f1_score"].append(f1)
            stats["ece"].append(ece)
    # Convert lists to tensors for easier handling
    for key in stats:
        stats[key] = torch.tensor(stats[key])
    return stats


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


def compute_bin_stats(
    predictions: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute bin statistics for reliability diagram.

    Divides the prediction probabilities into `n_bins` equal-width bins and computes
    the accuracy and confidence for each bin.

    Args:
        predictions: Tensor of shape (num_samples, num_classes) with predicted probabilities.
        targets: Tensor of shape (num_samples, num_classes) with binary ground truth labels.
        n_bins: Number of bins to divide the probability range [0, 1].
    Returns:
        bin_confidences: Tensor of shape (n_bins,) with average confidence per bin.
        bin_accuracies: Tensor of shape (n_bins,) with accuracy per bin.
        bin_counts: Tensor of shape (n_bins,) with number of samples
            in each bin.
    """
    # Convert to torch tensors if needed and ensure correct dtypes
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    else:
        predictions = predictions.float()  # Ensure predictions are float

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    else:
        targets = targets.float()  # Ensure targets are float for computation

    # Flatten for multilabel case - treat all predictions independently
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Create uniform bin edges
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)

    # Assign predictions to bins
    bin_indices = torch.bucketize(predictions_flat, bin_edges, right=False) - 1
    bin_indices = bin_indices.clamp(0, n_bins - 1)

    # Calculate bin statistics
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not mask.any():
            continue

        bin_preds = predictions_flat[mask]
        bin_targets = targets_flat[mask]

        if len(bin_preds) == 0:
            continue

        # Average confidence (predicted probability) in this bin
        avg_confidence = bin_preds.mean().item()

        # Fraction of positive outcomes (accuracy) in this bin
        accuracy = bin_targets.float().mean().item()

        # Number of samples in this bin
        count = len(bin_preds)

        bin_confidences.append(avg_confidence)
        bin_accuracies.append(accuracy)
        bin_counts.append(count)

    if not bin_confidences:
        # No valid bins, just plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # Convert to arrays for plotting
    bin_confidences = torch.tensor(bin_confidences)
    bin_accuracies = torch.tensor(bin_accuracies)
    bin_counts = torch.tensor(bin_counts)
    return bin_confidences, bin_accuracies, bin_counts
