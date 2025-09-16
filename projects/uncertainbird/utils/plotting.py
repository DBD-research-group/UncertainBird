from matplotlib import pyplot as plt
import numpy as np
import torch


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
    predictions, targets, n_bins=10, ax=None, title="Reliability Diagram"
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

    # Convert to torch tensors if needed
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

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
        "r-",
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
