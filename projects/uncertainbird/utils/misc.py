import os
import pickle
from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
import torch
from typing import Union, Tuple, Dict, Any, List


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


def load_data(
    log_dir: Union[str, Path],
) -> Tuple[Dict[str, Dict[str, Any]], torch.Tensor, torch.Tensor, List[str]]:
    data = {}
    dataset_names = sorted(
        [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    )

    # colormap without red (tab10 index 3 is red)
    colors = plt.cm.tab10
    skip_index = 3
    available_indices = [i for i in range(colors.N) if i != skip_index]
    dataset_colors = {}

    for i, ds in enumerate(dataset_names):
        data[ds] = {}
        color_idx = available_indices[i % len(available_indices)]
        dataset_colors[ds] = colors(color_idx)

        ds_path = os.path.join(log_dir, ds)
        pkl_files = [f for f in os.listdir(ds_path) if f.endswith(".pkl")]
        if not pkl_files:
            continue
        # pick most recent file
        pkl_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(ds_path, f)), reverse=True
        )
        file_path = os.path.join(ds_path, pkl_files[0])
        preds, t, metadata = load_dump(file_path)
        data[ds]["predictions"] = preds
        data[ds]["targets"] = t.int()
        data[ds]["metadata"] = metadata
        data[ds]["color"] = dataset_colors[ds]

    # concatenate
    valid_keys = [
        k
        for k, v in data.items()
        if isinstance(v, dict)
        and "predictions" in v
        and "targets" in v
        and isinstance(v["predictions"], torch.Tensor)
        and isinstance(v["targets"], torch.Tensor)
    ]

    if not valid_keys:
        raise ValueError("No datasets with both 'predictions' and 'targets' present.")

    # optionally report skipped datasets
    skipped = [k for k in data.keys() if k not in valid_keys]
    if skipped:
        print("Skipped datasets (missing predictions/targets):", skipped)

    predictions = torch.cat([data[k]["predictions"] for k in valid_keys], dim=0)
    targets = torch.cat([data[k]["targets"] for k in valid_keys], dim=0)

    data = {k: data[k] for k in valid_keys}

    return data, predictions, targets, valid_keys


def prune_non_target_classes(
    data: Dict[str, Dict[str, Any]], targets
) -> Dict[str, Dict[str, Any]]:
    """Remove classes that have no positive samples in any dataset.

    This function scans through all datasets and identifies classes that have at least one positive
    sample in the targets. It then prunes classes that have no positive samples across all datasets,
    updating both predictions and targets accordingly.

    Args:
        data: Dictionary where each key is a dataset name and each value is another dictionary
              containing 'predictions' and 'targets' tensors.
    Returns:
        A new dictionary with the same structure as the input, but with classes that have no
        positive samples removed from both predictions and targets.
    Example:
        >>> pruned_data = prune_non_target_classes(data)
        >>> for ds, content in pruned_data.items():
        ...     print(f"{ds}: {content['predictions'].shape}, {content['targets'].shape}")
    """

    for key in list(data.keys()):
        # keep the dict structure; only replace the predictions/targets tensors
        preds = data[key]["predictions"][:, targets.sum(dim=0).gt(0)]
        tars = data[key]["targets"][:, targets.sum(dim=0).gt(0)]
        data[key]["predictions"] = preds
        data[key]["targets"] = tars
    return data


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
    from uncertainbird.modules.metrics.uncertainty import (
        binary_miscalibration_score,
    )

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
        "mcs": [],
        "ece_under": [],
        "ece_over": [],
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
            mcs = binary_miscalibration_score(
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
            stats["mcs"].append(mcs)
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
        # No valid bins; return empty tensors
        return (
            torch.tensor([], dtype=predictions.dtype),
            torch.tensor([], dtype=predictions.dtype),
            torch.tensor([], dtype=predictions.dtype),
        )

    # Convert to arrays for plotting
    bin_confidences = torch.tensor(bin_confidences)
    bin_accuracies = torch.tensor(bin_accuracies)
    bin_counts = torch.tensor(bin_counts)
    return bin_confidences, bin_accuracies, bin_counts


def split_test_set(data, split_ratio=0.1):
    for base_name in list(data.keys()):
        if base_name.endswith("_cal") or base_name.endswith("_test"):
            continue
        entry = data[base_name]
        if "predictions" not in entry or "targets" not in entry:
            continue

        preds = entry["predictions"]
        tars = entry["targets"]
        n = preds.shape[0]
        if n == 0:
            continue

        cal_size_local = max(1, int(split_ratio * n))
        cal_idx_local = torch.arange(cal_size_local)
        test_idx_local = torch.arange(cal_size_local, n)

        # Adjust (copy) metadata to reflect subset size if present
        meta = dict(entry.get("metadata", {}))
        meta["total_samples"] = n

        # Create calibration split
        data[f"{base_name}_cal"] = {
            "predictions": preds[cal_idx_local],
            "targets": tars[cal_idx_local],
            "metadata": meta,
            "color": entry.get("color"),
        }

        # Create test split
        data[f"{base_name}_test"] = {
            "predictions": preds[test_idx_local],
            "targets": tars[test_idx_local],
            "metadata": meta,
            "color": entry.get("color"),
        }
    return data


def split_based_on_x_samples_per_class(data, samples_per_class=10):
    """Create calibration/test splits ensuring at most X positive samples per class.

    For every base dataset key (not already suffixed with ``_cal`` or ``_test``),
    this function scans samples in their existing order and assigns a sample to the
    calibration subset if it contains at least one positive class whose current
    calibration count is below ``samples_per_class``. Otherwise the sample is sent
    to the test subset. Negative-only samples (all-zero targets) go straight to test
    unless calibration would otherwise remain empty and there are no positives at all.

    Args:
        data (Dict[str, Dict[str, Any]]): Mapping of dataset names to dictionaries with
            at minimum keys ``predictions`` (Tensor[N, C]) and ``targets`` (Tensor[N, C]).
        samples_per_class (int): Maximum number of positive samples per class allowed in
            the calibration subset.

    Returns:
        Dict[str, Dict[str, Any]]: The input dictionary augmented with ``*_cal`` and
        ``*_test`` entries per processed dataset.
    """

    if samples_per_class <= 0:
        return data

    base_keys = [
        k for k in list(data.keys()) if not (k.endswith("_cal") or k.endswith("_test"))
    ]

    for base_name in base_keys:
        entry = data.get(base_name, {})
        preds = entry.get("predictions")
        tars = entry.get("targets")
        if not isinstance(preds, torch.Tensor) or not isinstance(tars, torch.Tensor):
            continue
        if preds.shape[0] != tars.shape[0]:
            continue
        N, C = tars.shape
        if N == 0:
            continue

        pos_counts = torch.zeros(C, dtype=torch.long)
        cal_indices = []
        test_indices = []

        for idx in range(N):
            sample_targets = tars[idx]
            if sample_targets.sum() == 0:
                test_indices.append(idx)
                continue
            positive_classes = (sample_targets > 0).nonzero(as_tuple=False).flatten()
            quota_mask = pos_counts[positive_classes] < samples_per_class
            if quota_mask.any():
                cal_indices.append(idx)
                for cls in positive_classes[quota_mask]:
                    pos_counts[cls] += 1
            else:
                test_indices.append(idx)

        if not cal_indices and tars.sum() > 0:
            # move first positive sample if calibration is empty
            first_positive = int((tars.sum(dim=1) > 0).nonzero(as_tuple=False)[0])
            cal_indices.append(first_positive)
            if first_positive in test_indices:
                test_indices.remove(first_positive)

        cal_idx_tensor = torch.tensor(cal_indices, dtype=torch.long)
        test_idx_tensor = torch.tensor(test_indices, dtype=torch.long)

        meta = dict(entry.get("metadata", {}))
        meta["total_samples"] = N
        color = entry.get("color")

        data[f"{base_name}_cal"] = {
            "predictions": preds[cal_idx_tensor] if len(cal_idx_tensor) else preds[:0],
            "targets": tars[cal_idx_tensor] if len(cal_idx_tensor) else tars[:0],
            "metadata": meta,
            "color": color,
        }
        data[f"{base_name}_test"] = {
            "predictions": (
                preds[test_idx_tensor] if len(test_idx_tensor) else preds[:0]
            ),
            "targets": tars[test_idx_tensor] if len(test_idx_tensor) else tars[:0],
            "metadata": meta,
            "color": color,
        }

    return data
