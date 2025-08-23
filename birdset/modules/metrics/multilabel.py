import torch
import torchmetrics
from torchmetrics.classification.average_precision import MultilabelAveragePrecision
from torchmetrics import Metric


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
            }
        )


class cmAP5(Metric):
    def __init__(
        self,
        num_labels: int,
        sample_threshold: int,
        thresholds=None,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_labels = num_labels
        self.sample_threshold = sample_threshold
        self.thresholds = thresholds

        self.multilabel_ap = MultilabelAveragePrecision(
            average="macro", num_labels=self.num_labels, thresholds=self.thresholds
        )

        # State variable to accumulate predictions and labels across batches
        self.add_state("accumulated_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("accumulated_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        # Accumulate predictions and labels
        self.accumulated_predictions.append(logits)
        self.accumulated_labels.append(labels)

    def compute(self) -> torch.Tensor:
        # Ensure that accumulated variables are lists
        if not isinstance(self.accumulated_predictions, list):
            self.accumulated_predictions = [self.accumulated_predictions]
        if not isinstance(self.accumulated_labels, list):
            self.accumulated_labels = [self.accumulated_labels]

        # Concatenate accumulated predictions and labels along the batch dimension
        all_predictions = torch.cat(self.accumulated_predictions, dim=0)
        all_labels = torch.cat(self.accumulated_labels, dim=0)

        # self.accumulated_predictions.clear()
        # self.accumulated_labels.clear()

        # Calculate class-wise AP
        class_aps = self.multilabel_ap(all_predictions, all_labels)

        if self.sample_threshold > 1:
            mask = all_labels.sum(axis=0) >= self.sample_threshold
            class_aps = torch.where(mask, class_aps, torch.nan)

        # Compute macro AP by taking the mean of class-wise APs, ignoring NaNs
        macro_cmap = torch.nanmean(class_aps)
        return macro_cmap

    # def reset(self):
    #     # Reset accumulated predictions and labels
    #     self.accumulated_predictions = []
    #     self.accumulated_labels = []


class cmAP(MultilabelAveragePrecision):
    def __init__(self, num_labels, thresholds=None):
        super().__init__(num_labels=num_labels, average="macro", thresholds=thresholds)

    def __call__(self, logits, labels):
        macro_cmap = super().__call__(logits, labels)
        return macro_cmap


class mAP(MultilabelAveragePrecision):

    def __init__(self, num_labels, thresholds=None):
        super().__init__(num_labels=num_labels, average="micro", thresholds=thresholds)

    def __call__(self, logits, labels):
        micro_cmap = super().__call__(logits, labels)
        return micro_cmap


class pcmAP(MultilabelAveragePrecision):
    # https://www.kaggle.com/competitions/birdclef-2023/overview/evaluation
    def __init__(
        self,
        num_labels: int,
        padding_factor: int = 5,
        average: str = "macro",
        thresholds=None,
        **kwargs
    ):

        super().__init__(
            num_labels=num_labels, average=average, thresholds=thresholds, **kwargs
        )

        self.padding_factor = padding_factor

    def __call__(self, logits, targets, **kwargs):
        ones = torch.ones(self.padding_factor, logits.shape[1])  # solve cuda!
        logits = torch.cat((logits, ones), dim=0)
        targets = torch.cat((targets, ones.int()), dim=0)
        pcmap = super().__call__(logits, targets, **kwargs)
        return pcmap

import torch
from torchmetrics import Metric

class MultilabelECEMacro(Metric):
    """
    Macro multi-label ECE that keeps all floating-point state in the model's dtype.

    For each bin k over [0,1]:
      mu_k           = mean predicted prob of all (i,c) in bin k
      prec_macro_k   = average over classes of precision_c(k) computed INSIDE the bin
      w_k            = |B_k| / (N*C)
    ECE = sum_k w_k * |prec_macro_k - mu_k|
    """
    full_state_update = False

    def __init__(self, num_labels: int, bins: int = 10, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        self.bins = int(bins)
        self.threshold = float(threshold)

        # States; dtypes/devices will be adapted to model outputs on first update()
        self.add_state("bin_sum_p", default=torch.zeros(self.bins, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("bin_count", default=torch.zeros(self.bins, dtype=torch.long),     dist_reduce_fx="sum")
        self.add_state("bin_tp",    default=torch.zeros(self.bins, self.num_labels, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("bin_fp",    default=torch.zeros(self.bins, self.num_labels, dtype=torch.long), dist_reduce_fx="sum")

    @torch.no_grad()
    def _ensure_device_dtype(self, probs: torch.Tensor):
        """Move/cast internal buffers to match probs' device/dtype."""
        d, dt = probs.device, probs.dtype
        if self.bin_sum_p.device != d or self.bin_sum_p.dtype != dt:
            self.bin_sum_p = self.bin_sum_p.to(device=d, dtype=dt)
        # long tensors: move device, keep dtype long
        if self.bin_count.device != d:
            self.bin_count = self.bin_count.to(device=d)
        if self.bin_tp.device != d:
            self.bin_tp = self.bin_tp.to(device=d)
        if self.bin_fp.device != d:
            self.bin_fp = self.bin_fp.to(device=d)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        probs: [N, C] probabilities in [0,1] (same dtype/device as model outputs; sigmoid outside)
        targets: [N, C] in {0,1}

        """

        if probs.ndim > 2:
            N, C = probs.shape[:2]
            probs   = probs.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)
            targets = targets.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)

        if not torch.isfinite(probs).all():
            bad = probs[~torch.isfinite(probs)]
            raise ValueError(f"[mECE] Non-finite probabilities detected (example: {bad.flatten()[:5].tolist()})")

        # Auto-sigmoid if looks like logits (mirror AUROC)
        if (probs.min() < 0) or (probs.max() > 1):
            probs = probs.sigmoid()

        probs = probs.clamp_(0.0, 1.0).nan_to_num_(nan=0.5, posinf=1.0, neginf=0.0)

        # Targets must be binary
        t_unique = torch.unique(targets)
        if not torch.all((t_unique == 0) | (t_unique == 1)):
            raise ValueError(f"[mECE] Targets must be binary {{0,1}}; got unique values: {t_unique.tolist()}")
        assert probs.ndim == 2 and probs.shape == targets.shape, "expected [N, C] probs/targets"
        self._ensure_device_dtype(probs)

        # Build edges in same dtype/device as probs; last bin right-inclusive via clamp below
        edges = torch.linspace(0.0, 1.0, self.bins + 1, device=probs.device, dtype=probs.dtype)

        targets = targets.detach().to(device=probs.device).bool()
        probs   = probs.detach()

        # Bin indices: [e_k, e_{k+1}); clamp p==1 to last bin
        bin_idx = torch.bucketize(probs, edges, right=False) - 1  # [-1..K-1]
        bin_idx.clamp_(0, self.bins - 1)

        # Flatten for index_add_
        flat_bins  = bin_idx.reshape(-1)                      # long
        flat_probs = probs.reshape(-1)                        # model dtype

        # Accumulate mean prob stats (dtypes match)
        self.bin_sum_p.index_add_(0, flat_bins, flat_probs)
        self.bin_count.index_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=torch.long))

        # Precision components inside each bin at threshold
        pred_pos = probs >= torch.as_tensor(self.threshold, device=probs.device, dtype=probs.dtype)
        for k in range(self.bins):
            in_k = (bin_idx == k)
            if not in_k.any():
                continue
            pp_k = pred_pos & in_k
            self.bin_tp[k] += (pp_k & targets).sum(dim=0)          # long
            self.bin_fp[k] += (pp_k & (~targets)).sum(dim=0)       # long

    @torch.no_grad()
    def _per_bin_stats(self):
        """
        Returns:
          bin_mu: [K] mean predicted prob per bin (model dtype)
          bin_prec_macro: [K] macro precision per bin (model dtype)
          bin_mass: [K] fraction of (i,c) in each bin (model dtype)
        """
        dt = self.bin_sum_p.dtype
        # Mean prob per bin
        bin_mu = torch.zeros_like(self.bin_sum_p, dtype=dt)
        non_empty = self.bin_count > 0
        bin_mu[non_empty] = self.bin_sum_p[non_empty] / self.bin_count[non_empty].to(dtype=dt)

        # Macro precision per bin
        denom = self.bin_tp + self.bin_fp  # long
        prec_c = torch.zeros_like(denom, dtype=dt)
        mask = denom > 0
        prec_c[mask] = self.bin_tp[mask].to(dtype=dt) / denom[mask].to(dtype=dt)
        bin_prec_macro = prec_c.mean(dim=1).to(dtype=dt)

        # Bin mass
        total_pairs = self.bin_count.sum().clamp_min(1).to(dtype=dt)
        bin_mass = self.bin_count.to(dtype=dt) / total_pairs
        return bin_mu, bin_prec_macro, bin_mass

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        bin_mu, bin_prec_macro, bin_mass = self._per_bin_stats()
        # print("bin_count:", self.bin_count)
        # print("bin_sum_p:", self.bin_sum_p)
        # print("bin_tp:", self.bin_tp.sum(dim=1))
        # print("bin_fp:", self.bin_fp.sum(dim=1))
        return (bin_mass * (bin_prec_macro - bin_mu).abs()).sum()

    @torch.no_grad()
    def compute_with_details(self):
        ece = self.compute()
        bin_mu, bin_prec_macro, bin_mass = self._per_bin_stats()
        return ece, bin_mu, bin_prec_macro, bin_mass

    

# class TopKAccuracy(torchmetrics.Metric):
#     def __init__(self, topk=1, **kwargs):
#         super().__init__(**kwargs)
#         self.topk = topk
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#     def update(self, preds, targets):
#         # Get the top-k predictions
#         _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
#         targets = targets.to(topk_pred_indices.device)

#         # Convert one-hot encoded targets to class indices
#         target_indices = targets.argmax(dim=1)

#         # Compare each of the top-k indices with the target index
#         correct = topk_pred_indices.eq(target_indices.unsqueeze(1)).any(dim=1)

#         # Update correct and total
#         self.correct += correct.sum()
#         self.total += targets.size(0)

#     def compute(self):
#         return self.correct.float() / self.total

# class TopKAccuracy(torchmetrics.Metric):
#     def __init__(self, topk=1, **kwargs):
#         super().__init__(**kwargs)
#         self.topk = topk
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, preds, targets):
#         # top-k predictions
#         _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
#         targets = targets.to(topk_pred_indices.device)

#         #expand targets to match the shape of topk_pred_indices for broadcasting
#         expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)

#         #check if any of the top-k predictions match the true labels
#         correct = expanded_targets.gather(2, topk_pred_indices.unsqueeze(-1)).any(dim=1)

#         self.correct += correct.sum()
#         self.total += targets.size(0)

#     def compute(self):
#         return self.correct.float() / self.total


class TopKAccuracy(torchmetrics.Metric):
    def __init__(self, topk=1, include_nocalls=False, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        # Get the top-k predictions
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
        targets = targets.to(preds.device)
        no_call_targets = targets.sum(dim=1) == 0

        # consider no_call instances (a threshold is needed here!)
        if self.include_nocalls:
            # check if top-k predictions for all-negative instances are less than threshold
            no_positive_predictions = (
                preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            )
            correct_all_negative = no_call_targets & no_positive_predictions.all(dim=1)

        else:
            # no_calls are removed, set to 0
            correct_all_negative = torch.tensor(0).to(targets.device)

        # convert one-hot encoded targets to class indices for positive cases
        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        correct_positive = expanded_targets.gather(
            2, topk_pred_indices.unsqueeze(-1)
        ).any(dim=1)

        # update correct and total, excluding all-negative instances if specified
        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            self.total += targets.size(0)

    def compute(self):
        return self.correct.float() / self.total
