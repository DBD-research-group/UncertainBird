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


def _bin_stats(predictions, targets, n_bins=10, quantile=False):
    """
    Compute bin-wise calibration statistics for reliability diagrams and ECE/MCE.

    This function bins predictions and targets (flattened) into either uniform or quantile bins,
    then computes, for each bin:
      - The mean predicted confidence (average predicted probability)
      - The mean accuracy (fraction of positives)
      - The weight (fraction of total samples in the bin)
      - The calibration gap (absolute difference between accuracy and confidence)
    It then aggregates these to compute:
      - ECE (Expected Calibration Error): weighted average of calibration gaps
      - MCE (Maximum Calibration Error): largest calibration gap across bins

    Args:
        predictions (array-like): Model predicted probabilities, shape (N,) or (N, C). Flattened to 1D.
        targets (array-like): Ground truth binary labels, same shape as predictions. Flattened to 1D.
        n_bins (int): Number of bins to use for calibration statistics.
        quantile (bool): If True, use quantile bins (equal number of samples per bin). If False, use uniform bins.

    Returns:
        bin_confs (np.ndarray): Mean predicted confidence per bin, shape (n_bins,)
        bin_accs (np.ndarray): Mean accuracy per bin, shape (n_bins,)
        bin_weights (np.ndarray): Fraction of samples in each bin, shape (n_bins,)
        ece (float): Expected Calibration Error (weighted average of bin gaps)
        mce (float): Maximum Calibration Error (largest bin gap)

    Notes:
        - Binning is performed on the flattened predictions, treating all predictions as independent.
        - For quantile binning, bin edges are chosen so each bin has (approximately) equal number of samples.
        - For uniform binning, bin edges are equally spaced in [0, 1].
        - If a bin is empty, its confidence is set to the midpoint of the bin, accuracy is NaN, weight and gap are 0.
        - ECE is the sum over bins of (bin_weight * bin_gap), ignoring empty bins.
        - MCE is the maximum bin_gap over non-empty bins.

    Example:
        >>> preds = np.array([0.1, 0.4, 0.8, 0.9])
        >>> targets = np.array([0, 0, 1, 1])
        >>> _bin_stats(preds, targets, n_bins=2)
        (array([0.25, 0.85]), array([0., 1.]), array([0.5, 0.5]), 0.15, 0.15)
    """
    # Flatten predictions and targets to 1D arrays
    conf = np.asarray(predictions).reshape(-1)
    labels = np.asarray(targets).reshape(-1)
    n = conf.size

    # Choose bin edges: quantile or uniform
    if quantile:
        qs = np.linspace(0, 1, n_bins + 1)
        try:
            # Newer numpy: method argument for quantile
            edges = np.quantile(conf, qs, method="linear")
        except TypeError:
            # Older numpy: no method argument
            edges = np.quantile(conf, qs)
        # Ensure edges are exactly [0, 1] at endpoints
        edges[0], edges[-1] = 0.0, 1.0
        # Ensure edges are non-decreasing (can happen with repeated values)
        edges = np.maximum.accumulate(edges)
    else:
        # Uniform bins in [0, 1]
        edges = np.linspace(0, 1, n_bins + 1)

    bin_confs, bin_accs, bin_weights, bin_gaps = [], [], [], []
    for b in range(n_bins):
        left, right = edges[b], edges[b + 1]
        # For all but last bin: [left, right)
        # For last bin: [left, right] (include right endpoint)
        if b < n_bins - 1:
            m = (conf >= left) & (conf < right)
        else:
            m = (conf >= left) & (conf <= right)
        if np.any(m):
            # Compute mean confidence and accuracy for this bin
            c_mean = conf[m].mean()
            a_mean = labels[m].mean()
            w = m.sum() / n  # Fraction of total samples in this bin
            gap = abs(a_mean - c_mean)  # Calibration gap
        else:
            # Empty bin: set confidence to bin midpoint, accuracy NaN, weight/gap 0
            c_mean = (left + right) / 2
            a_mean, w, gap = np.nan, 0.0, 0.0
        bin_confs.append(c_mean)
        bin_accs.append(a_mean)
        bin_weights.append(w)
        bin_gaps.append(gap)

    # Convert lists to arrays
    bin_confs, bin_accs, bin_weights, bin_gaps = map(
        np.array, [bin_confs, bin_accs, bin_weights, bin_gaps]
    )
    # Only consider bins with nonzero weight for ECE/MCE
    valid = bin_weights > 0
    ece = float(np.sum(bin_weights[valid] * bin_gaps[valid])) if np.any(valid) else 0.0
    mce = float(np.max(bin_gaps[valid])) if np.any(valid) else 0.0
    return bin_confs, bin_accs, bin_weights, ece, mce


def _plot_reliability(bin_confs, bin_accs, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    mask = ~np.isnan(bin_accs)
    ax.plot(bin_confs[mask], bin_accs[mask], marker="o", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    return ax


def _get_k(predictions, targets, k=5):
    N, T = predictions.shape
    k = min(k, T)
    idx = np.argpartition(predictions, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(N)[:, None]
    conf_topk = predictions[rows, idx].reshape(-1)
    labels_topk = targets[rows, idx].reshape(-1)
    return conf_topk, labels_topk
