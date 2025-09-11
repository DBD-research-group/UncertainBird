import numpy as np
from uncertainbird.utils.plotting import plot_reliability_diagram


def ece(preds, labels, n_bin=10, mode="l1", savepath=False):
    bin_preds, bin_count, bin_total, bins = calibration_summary(
        preds, labels, "uniform", n_bin=n_bin
    )
    prob_pred = np.array([elem.mean() if len(elem) > 0 else 0.0 for elem in bin_preds])
    prob_data = np.zeros(len(bin_total))
    prob_data[bin_total != 0] = bin_count[bin_total != 0] / bin_total[bin_total != 0]

    val = 0
    if mode == "l1":
        val = np.sum(np.abs(prob_data - prob_pred) * bin_total) / np.sum(bin_total)
    elif mode == "l2":
        val = np.sum((np.abs(prob_data - prob_pred) ** 2) * bin_total) / np.sum(
            bin_total
        )
    elif mode == "inf":
        val = np.max(np.abs(prob_data - prob_pred))
    else:
        assert False, "no correct mode specified: (l1, l2, inf)"

    if savepath != False:
        plot_reliability_diagram(prob_pred, prob_data, bin_total, preds, bins, savepath)

    return val


def calibration_summary(
    preds, labels, strategy="pavabc", n_min=10, n_max=1000, n_bin=10
):
    """Summarize prediction calibration by binning probability scores.

    This is a helper that routes to different binning strategies and returns
    per-bin aggregates needed for reliability diagrams and Expected Calibration
    Error (ECE) style metrics.

    Supported strategies:
      * 'pavabc'  : Adaptive isotonic binning using constrained PAVA (see _pavabc) with
                    minimum (n_min) and maximum (n_max) bin size constraints. Produces
                    variable-width bins that are monotonic in empirical accuracy.
      * 'pava'    : Pure isotonic binning (no explicit size constraints) by calling
                    _pavabc with n_min=0 and n_max=len(preds)+1.
      * 'uniform' : Fixed, equally spaced probability intervals (n_bin bins).
      * 'quantile': Bins formed so each contains ~equal numbers of samples (n_bin bins).

    Args:
        preds (np.ndarray): 1-D array of predicted probabilities in [0,1].
        labels (np.ndarray): 1-D array of binary labels {0,1} aligned with preds.
        strategy (str): Binning strategy ('pavabc', 'pava', 'uniform', 'quantile').
        n_min (int): Minimum bin size for 'pavabc'. Ignored by other strategies.
        n_max (int): Maximum provisional merged bin size for 'pavabc'. Ignored by others.
        n_bin (int): Number of bins for 'uniform' and 'quantile' strategies.

    Returns:
        bin_preds (List[np.ndarray]): List of per-bin prediction arrays.
        bin_count (np.ndarray): Count of positive labels per bin.
        bin_total (np.ndarray): Total number of samples per bin.
        bins (np.ndarray): Bin edges of length (#bins + 1).

    Notes:
        * All assertions ensure inputs are valid probability / binary arrays.
        * The caller can compute empirical bin accuracy as bin_count / bin_total
          (guarding against division by zero for empty bins in sparse scenarios).
        * For ECE (l1) one typically computes: sum_w |acc - conf| with weights = bin_total / N,
          where conf is mean(predictions) in the bin.

    Example:
        >>> bin_preds, bin_count, bin_total, bins = calibration_summary(p, y, 'uniform', n_bin=15)
        >>> acc = bin_count / np.maximum(1, bin_total)
        >>> conf = np.array([bp.mean() if len(bp) else 0.0 for bp in bin_preds])
    """
    assert np.all(preds >= 0.0) and np.all(
        preds <= 1.0
    ), "Prediction Out of Range [0, 1]"
    assert np.all((labels == 0) | (labels == 1)), "Label Not 0 or 1"

    if strategy == "pavabc":
        bin_preds, bin_count, bin_total, bins = _pavabc(
            preds, labels, n_min=n_min, n_max=n_max
        )
    elif strategy == "pava":
        bin_preds, bin_count, bin_total, bins = _pavabc(
            preds, labels, n_min=0, n_max=len(preds) + 1
        )
    elif strategy == "uniform":
        bin_preds, bin_count, bin_total, bins = _calibration_process(
            preds, labels, strategy, n_bin
        )
    elif strategy == "quantile":
        bin_preds, bin_count, bin_total, bins = _calibration_process(
            preds, labels, strategy, n_bin
        )
    else:
        assert False, "no correct strategy specified: (uniform, quantile, pava, ncpave)"

    return bin_preds, bin_count, bin_total, bins


def _pavabc(x, y, n_min=0, n_max=10000):
    """Perform isotonic (monotonic non-decreasing) binning using PAVA with bin size constraints.

    This implements a *Pool Adjacent Violators Algorithm* (PAVA) variant that additionally
    enforces lower (``n_min``) and upper (``n_max``) constraints on the number of samples per
    final bin. The procedure is used to derive *adaptive calibration bins* for reliability
    diagrams / calibration error computation.

    High-level steps:
      1. Sort predictions ``x`` (probabilities) and reorder labels ``y`` accordingly.
      2. Initialize each observation as its own provisional bin (weight = 1, sum of labels = y_i).
      3. Iterate forward; whenever two adjacent bins (previous, current) would violate
         isotonicity (i.e. previous mean >= current mean) OR their combined size is below
         ``n_min`` (must merge) OR (their combined size <= ``n_max`` AND isotonicity violated),
         merge them (add counts and label sums) and keep checking recursively (classic PAVA
         pooling step with number constraints from Jiang et al. style adaptive binning).
      4. (If ``n_min`` > 0) ensure the trailing remainder of samples also satisfies the minimum
         size requirement by merging the last ``n_min`` samples.
      5. Construct final bin boundaries as midpoints between adjacent pooled segments, yielding a
         vector of length (#bins + 1) that always starts at 0.0 and ends at 1.0.

    Rationale for constraints:
      * ``n_min`` forces a minimum occupancy, preventing extremely small bins that would cause
        high-variance empirical accuracy estimates.
      * ``n_max`` limits how large a bin may become when enforcing monotonicity, maintaining
        resolution where data are dense.

    Parameters
    ----------
    x : array-like (N,)
        Prediction scores already in [0, 1]. They are *not* modified; only their ordering matters.
    y : array-like (N,)
        Binary labels {0,1} corresponding to ``x``.
    n_min : int, default 0
        Minimum allowed number of samples in any *final* bin. If > 0, the last ``n_min`` samples
        are merged (if possible) to satisfy the size constraint.
    n_max : int, default 10000
        Maximum allowed number of samples when deciding whether two adjacent bins may be merged
        (condition2). Acts as a soft upper bound; bins can exceed it only via the final merge from
        the trailing remainder step.

    Returns
    -------
    bin_preds : list[np.ndarray]
        List of prediction arrays for each bin (the raw predictions falling into that bin).
    bin_count : np.ndarray (B,)
        Sum of positive labels per bin (i.e. numerator for empirical accuracy).
    bin_total : np.ndarray (B,)
        Total number of samples per bin (i.e. denominator for empirical accuracy / weight).
    bins : np.ndarray (B+1,)
        Bin edge locations in probability space: [0.0, e_1, ..., e_{B-1}, 1.0]. Midpoints are used
        between adjacent pooled segments; edges are inclusive of the left boundary and exclusive
        of the right, except for the final bin.

    Notes
    -----
    * Complexity is O(N) after the initial O(N log N) sort.
    * The resulting bins are isotonic in their empirical positive rate (bin_count / bin_total).
    * This function is analogous to adaptive / isotonic binning used for calibration plots,
      differing from *uniform* or *quantile* binning which ignore monotonicity.
    """
    ### Sort (Start) ###
    order = np.argsort(x)
    xsort = x[order]
    ysort = y[order]
    num_y = len(ysort)
    ### Sort (End) ###

    def _condition(y0, y1, w0, w1):
        # Decide whether to merge the previous bin (sum y0, weight w0) with the current bin
        # (sum y1, weight w1). Merge if:
        #  (1) Combined size still below minimum (mandatory merge), OR
        #  (2) Combined size does not exceed maximum AND isotonicity is violated
        #      (previous mean >= current mean).
        condition1 = w0 + w1 <= n_min  # enforce minimum size
        condition2 = w0 + w1 <= n_max  # upper size constraint respected
        condition3 = y0 / w0 >= y1 / w1  # violates strict increasing means
        return condition1 or (condition2 and condition3)

    ### PAVA with Number Constraint (Start) ###
    count = -1
    iso_y = []  # list of summed labels per provisional bin
    iso_w = []  # list of counts per provisional bin
    for i in range(
        num_y - n_min
    ):  # leave a tail of n_min samples (handled later) if n_min>0
        count += 1
        iso_y.append(ysort[i])
        iso_w.append(1)
        # Merge backward while constraints dictate
        while count > 0 and _condition(
            iso_y[count - 1], iso_y[count], iso_w[count - 1], iso_w[count]
        ):
            iso_y[count - 1] += iso_y[count]
            iso_w[count - 1] += iso_w[count]
            iso_y.pop()
            iso_w.pop()
            count -= 1
    if n_min > 0:
        # Aggregate the final n_min samples (ensures last bin meets minimum size)
        count += 1
        iso_y.append(sum(ysort[num_y - n_min : num_y]))
        iso_w.append(n_min)
        # Optionally merge with previous if doing so still respects n_max
        if iso_w[-1] + n_min <= n_max and count > 0:
            iso_y[count - 1] += iso_y[count]
            iso_w[count - 1] += iso_w[count]
            iso_y.pop()
            iso_w.pop()
            count -= 1
    ### PAVA with Number Constraint (End) ###

    ### Process return values (Start) ###
    index = np.r_[
        0, np.cumsum(iso_w)
    ]  # cumulative indices delimiting bins in sorted arrays
    # Internal edges: midpoint between last element of previous bin and first of next bin
    bins = np.r_[
        0.0,
        [
            (xsort[index[j] - 1] + xsort[index[j]]) / 2.0
            for j in range(1, len(index) - 1)
        ],
        1.0,
    ]
    bin_count = np.array(iso_y)  # positives per bin
    bin_total = np.array(iso_w)  # total samples per bin
    bin_preds = [xsort[index[j] : index[j + 1]] for j in range(len(index) - 1)]
    ### Process return values (End) ###

    return bin_preds, bin_count, bin_total, bins


def _calibration_process(preds, labels, strategy="uniform", n_bin=10):
    """Bin predictions and labels according to a chosen calibration strategy.

    Produces per-bin aggregates needed to compute reliability diagrams and
    Expected Calibration Error style metrics. This function implements two
    deterministic binning schemes:

    Strategies
    ----------
    uniform :
        Fixed-width bins that partition [0,1] into ``n_bin`` equal intervals.
        Each prediction is assigned by simple digitization. Empty bins are
        allowed (bin_total can be 0).
    quantile :
        Variable-width bins containing (approximately) equal numbers of samples.
        Achieved by sorting predictions and slicing into ``n_bin`` consecutive
        chunks of (nearly) identical size. Guarantees no empty bins unless
        ``len(preds) < n_bin``.

    Args
    ----
    preds : np.ndarray (N,)
        Predicted probabilities in [0,1].
    labels : np.ndarray (N,)
        Binary ground-truth labels {0,1} aligned with ``preds``.
    strategy : str, default 'uniform'
        Either 'uniform' or 'quantile'.
    n_bin : int, default 10
        Number of bins to create.

    Returns
    -------
    bin_preds : list[np.ndarray]
        List of prediction arrays for each bin (raw predictions).
    bin_count : np.ndarray (B,)
        Number of positive labels per bin (sum of labels inside bin).
    bin_total : np.ndarray (B,)
        Total number of samples in each bin.
    bins : np.ndarray (B+1,)
        Bin edge array. For 'uniform', edges are equally spaced; for 'quantile'
        edges are midpoints between adjacent sorted predictions (with 0.0 and 1.0
        as global boundaries).

    Notes
    -----
    * For 'uniform', a temporary trick sets the last edge to >1 to ensure value
      1.0 falls into the final bin when digitizing.
    * Empirical bin accuracy can be computed as bin_count / bin_total (guard
      against division by zero for empty bins).
    * Mean predicted confidence per bin is e.g. np.array([bp.mean() if len(bp) else 0 for bp in bin_preds]).
    """
    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bin + 1)
        bins[-1] = 1.1  # trick to include 'pred=1.0' in the final bin during digitize
        indices = np.digitize(preds, bins, right=False) - 1
        bins[-1] = 1.0  # restore proper upper edge
        bin_count = np.array(
            [sum(labels[indices == i]) for i in range(bins.shape[0] - 1)]
        ).astype(int)
        bin_total = np.array(
            [len(labels[indices == i]) for i in range(bins.shape[0] - 1)]
        ).astype(int)
        bin_preds = [preds[indices == i] for i in range(bins.shape[0] - 1)]
        return bin_preds, bin_count, bin_total, bins

    elif strategy == "quantile":
        quantile = np.linspace(0, 1, n_bin + 1)
        sortedindices = np.argsort(preds)
        sortedlabels = labels[sortedindices]
        sortedpreds = preds[sortedindices]
        idpartition = (quantile * len(labels)).astype(int)
        bin_count = np.array(
            [sum(sortedlabels[s:e]) for s, e in zip(idpartition, idpartition[1:])]
        ).astype(int)
        bin_total = np.array(
            [len(sortedlabels[s:e]) for s, e in zip(idpartition, idpartition[1:])]
        ).astype(int)
        bin_preds = [sortedpreds[s:e] for s, e in zip(idpartition, idpartition[1:])]
        bins = np.array(
            [0.0]
            + [(sortedpreds[e - 1] + sortedpreds[e]) / 2.0 for e in idpartition[1:-1]]
            + [1.0]
        )
        return bin_preds, bin_count, bin_total, bins

    else:
        assert False, "no correct strategy specified: (uniform, quantile)"
