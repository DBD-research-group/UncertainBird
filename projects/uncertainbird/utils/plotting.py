from matplotlib import colors, pyplot as plt
import numpy as np
import torch

from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelAUROC,
)
from uncertainbird.utils.misc import compute_bin_stats
from birdset.modules.metrics import cmAP
from uncertainbird.modules.metrics.uncertainty import (
    MultilabelCalibrationError,
    TopKMultiLabelCalibrationError,
)


def plot_pr_curve(predictions, targets, ax=None):
    """Plot Precision-Recall curve in the multilabel setting."""
    from sklearn.metrics import precision_recall_curve, auc

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(
        targets.flatten(), predictions.flatten()
    )
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True)

    return ax


def plot_pr_curve_per_class(predictions, targets, class_names=None, ax=None):
    """Plot Precision-Recall curve for each class in a multi-class setting."""
    from sklearn.metrics import precision_recall_curve, auc

    num_classes = predictions.shape[1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))

    for class_idx in range(num_classes):
        class_preds = predictions[:, class_idx]
        class_targets = targets[:, class_idx]

        precision, recall, _ = precision_recall_curve(class_targets, class_preds)
        pr_auc = auc(recall, precision)

        label = (
            f"Class {class_idx} PR AUC = {pr_auc:.2f}"
            if class_names is None
            else f"{class_names[class_idx]} PR AUC = {pr_auc:.2f}"
        )
        ax.plot(recall, precision, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve per Class")
    ax.legend()
    ax.grid(True)

    return ax


def plot_reliability_diagram(
    predictions, targets, n_bins=10, ax=None, title="Reliability Diagram", color="red"
):
    """Plot a reliability diagram to visualize calibration of probabilistic predictions.

    The reliability diagram shows how well the predicted probabilities of a model
    correspond to the actual outcomes. Perfectly calibrated predictions would fall
    along the diagonal line from (0,0) to (1,1).

    Args:
        predictions: The predictions tensor of shape (num_samples, num_classes) with values in [0, 1].
        targets: The targets tensor of shape (num_samples, num_classes) with binary labels (0 or 1).
        n_bins: Number of bins to use for the reliability diagram.
        ax: Optional matplotlib axis to plot on. If None, a new figure and axis are created.
        title: Title for the plot.

    Returns:
        ax: The matplotlib axis containing the reliability diagram.

    Note:
        - Predictions are binned into `n_bins` equally spaced intervals.
        - For each bin, the average predicted probability and the fraction of positive outcomes are computed.
        - The diagram includes a reference diagonal line representing perfect calibration.

    Example:
        >>> plot_reliability_diagram(predictions, targets, n_bins=10)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    bin_confidences, bin_accuracies, bin_counts = compute_bin_stats(
        predictions, targets, n_bins=n_bins
    )

    # Plot reliability diagram
    # Size points by number of samples in each bin
    max_count = bin_counts.max().item()
    sizes = 50 + 200 * (bin_counts.float() / max_count)  # Scale point sizes

    scatter = ax.scatter(
        bin_confidences,
        bin_accuracies,
        s=sizes,
        alpha=0.7,
        c=bin_counts,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar to show sample counts
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Samples per bin')

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

    # Connect points to show the calibration curve
    ax.plot(
        bin_confidences,
        bin_accuracies,
        color=color,
        alpha=0.8,
        linewidth=2,
        label="Model Calibration",
    )

    # Set labels and title
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_combined_reliability_diagram(
    data, n_bins=10, title="Combined Reliability Diagram"
):
    """Plot a combined reliability diagram for multiple datasets.

    This function creates a reliability diagram that combines predictions and targets
    from multiple datasets. Each dataset is represented with a different color, allowing
    for easy comparison of calibration across datasets.

    Args:
        data: A dictionary where keys are dataset names and values are tuples of
              (predictions, targets, metadata, metrics). Predictions and targets should be tensors of shape
              (num_samples, num_classes).
        n_bins: Number of bins to use for the reliability diagram.
        title: Title for the plot.
    Returns:
        ax: The matplotlib axis containing the combined reliability diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for dataset in data:
        predictions = data[dataset]["predictions"]
        targets = data[dataset]["targets"]
        bin_confidences, bin_accuracies, bin_counts = compute_bin_stats(
            predictions, targets, n_bins=n_bins
        )

        # Plot reliability diagram for this dataset
        max_count = bin_counts.max().item()
        sizes = 50 + 200 * (bin_counts.float() / max_count)  # Scale point sizes

        scatter = ax.scatter(
            bin_confidences,
            bin_accuracies,
            s=sizes,
            alpha=0.7,
            c=[data[dataset]["color"]],
            edgecolors="black",
            linewidth=0.5,
            label=(
                f"{dataset} | ECE weighted: {data[dataset]['metrics']['ece_weighted']*100:.2f}"
                if "metrics" in data[dataset]
                else dataset
            ),
        )

        # Connect points to show the calibration curve
        ax.plot(
            bin_confidences,
            bin_accuracies,
            color=data[dataset]["color"],
            alpha=0.8,
            linewidth=2,
        )

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

    # Set labels and title
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_confidence_histogram(
    predictions, n_bins=10, ax=None, title="Confidence Histogram"
):
    """Plot a histogram of prediction confidences.

    This function visualizes the distribution of predicted probabilities (confidences)
    from a model. It helps to understand how confident the model is in its predictions.

    Args:
        predictions: The predictions tensor of shape (num_samples, num_classes) with values in [0, 1].
        n_bins: Number of bins to use for the histogram.
        ax: Optional matplotlib axis to plot on. If None, a new figure and axis are created.
        title: Title for the plot.

    Returns:
        ax: The matplotlib axis containing the confidence histogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to torch tensor if needed
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)

    # Flatten for multilabel case - treat all predictions independently
    predictions_flat = predictions.flatten()

    # Plot histogram
    ax.hist(
        predictions_flat.numpy(),
        bins=n_bins,
        range=(0.0, 1.0),
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )

    # Set labels and title
    ax.set_xlabel("Predicted Probability (Confidence)")
    ax.set_ylabel("Number of Predictions (Log Scale)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return ax


def plot_class_frequency(targets):
    """
    Plot class frequency distribution for multilabel classification data.

    Creates two visualizations:
    1. A histogram showing the distribution of classes binned by sample count
    2. A line plot showing sorted per-class sample counts

    Args:
        targets (torch.Tensor): A 2D tensor of shape (n_samples, n_classes)
            containing binary labels for multilabel classification. Each row
            represents a sample and each column represents a class, with 1
            indicating the presence of that class.

    Returns:
        None: Displays two matplotlib plots directly.

    Example:
        >>> import torch
        >>> # Create sample multilabel data (100 samples, 50 classes)
        >>> targets = torch.randint(0, 2, (100, 50))
        >>> plot_class_frequency(targets)

    Note:
        - Both plots use logarithmic y-axis scaling for better visualization
        - The histogram uses predefined bins: [0,1), [1,2), [2,3), [3,5),
          [5,10), [10,20), [20,50), [50,100), [100,200), [200,500),
          [500,1000), [1000,2000), [2000,5000), [5000,+∞)
        - Requires matplotlib and numpy to be imported
    """

    # Per-class sample counts (multilabel: sum over samples axis)
    class_counts = targets.sum(dim=0).cpu().numpy().astype(int)

    # Sort counts (optional diagnostic)
    sorted_counts = np.sort(class_counts)

    # Define frequency bins: adjust if needed
    bin_edges = np.array(
        [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    )

    # Histogram over bins
    hist, edges = np.histogram(class_counts, bins=bin_edges)

    # Build readable bin labels
    labels = []
    for i in range(len(edges) - 1):
        left = int(edges[i])
        right = edges[i + 1]
        if np.isinf(right):
            labels.append(f"{left}+")
        else:
            right = int(right) - 1
            if left == right:
                labels.append(f"{left}")
            else:
                labels.append(f"{left}-{right}")

    # Histogram (y on log scale)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(hist)), hist)
    plt.xticks(range(len(hist)), labels, rotation=45, ha="right")
    plt.ylabel("Number of classes (log scale)")
    plt.xlabel("Number of samples (per class)")
    plt.title("Class frequency distribution (binned)")
    plt.yscale("log")
    plt.tight_layout()

    # Sorted counts (already log scale)
    plt.figure(figsize=(10, 4))
    plt.plot(sorted_counts)
    plt.ylabel("Samples per class (log scale)")
    plt.xlabel("Class (sorted)")
    plt.title("Sorted per-class sample counts")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def print_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    num_labels = targets.shape[1]
    accuracy = MultilabelAccuracy(num_labels=num_labels)(predictions, targets)
    cmAP_metric = cmAP(num_labels=num_labels)(predictions, targets)
    precision = MultilabelPrecision(num_labels=num_labels)(predictions, targets)
    recall = MultilabelRecall(num_labels=num_labels)(predictions, targets)
    f1 = MultilabelF1Score(num_labels=num_labels)(predictions, targets)
    auroc = MultilabelAUROC(num_labels=num_labels)(predictions, targets)
    ece = MultilabelCalibrationError(n_bins=10)(predictions, targets)
    ece_weighted = MultilabelCalibrationError(n_bins=10, multilabel_average="weighted")(
        predictions, targets
    )

    criterion = "target class"  # "probability"  # "predicted class"  # "target class"
    ece_3 = TopKMultiLabelCalibrationError(k=3, n_bins=10, criterion=criterion)(
        predictions, targets
    )
    ece_5 = TopKMultiLabelCalibrationError(k=5, n_bins=10, criterion=criterion)(
        predictions, targets
    )
    ece_10 = TopKMultiLabelCalibrationError(k=10, n_bins=10, criterion=criterion)(
        predictions, targets
    )
    ece_21 = TopKMultiLabelCalibrationError(k=21, n_bins=10, criterion=criterion)(
        predictions, targets
    )

    metrics = {
        "accuracy": accuracy,
        "cmAP": cmAP_metric,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": auroc,
        "ece": ece,
        "ece_weighted": ece_weighted,
        "ece_top_3": ece_3,
        "ece_top_5": ece_5,
        "ece_top_10": ece_10,
        "ece_top_21": ece_21,
    }

    print("Accuracy:", accuracy)
    print("cmAP:", cmAP_metric)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUROC:", auroc)
    print("ECE:", ece)
    print("ECE Weighted:", ece_weighted)
    print("ECE Top-3:", ece_3)
    print("ECE Top-5:", ece_5)
    print("ECE Top-10:", ece_10)
    print("ECE Top-21:", ece_21)

    return metrics


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
