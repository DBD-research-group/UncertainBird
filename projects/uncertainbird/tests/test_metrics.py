import pytest
import torch
from uncertainbird.modules.metrics.uncertainty import compute_calibration_error
from uncertainbird.utils.plotting import _bin_stats


class TestMultilabelCalibrationError:
    def test_multilabel_calibration_error_example(self):
        preds = torch.tensor(
            [
                [0.25, 0.20, 0.55],
                [0.55, 0.05, 0.40],
                [0.10, 0.30, 0.60],
                [0.90, 0.05, 0.05],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]
        )
        l1 = compute_calibration_error(
            preds, target, n_bins=3, norm="l1", type="marginal"
        )
        print("Torch Metrics implementation ECE: ", l1)
        bin_confs, bin_accs, bin_weights, ece, mce = _bin_stats(
            preds, target, n_bins=3, quantile=False
        )
        print("XMLC ECE: ", ece)
        l2 = compute_calibration_error(
            preds, target, n_bins=3, norm="l2", type="marginal"
        )
        max_ = compute_calibration_error(
            preds, target, n_bins=3, norm="max", type="marginal"
        )
        assert torch.isclose(l1, torch.tensor(0.1917), atol=1e-4)
        assert torch.isclose(l2, torch.tensor(0.2075), atol=1e-4)
        assert torch.isclose(max_, torch.tensor(0.550), atol=1e-4)

    def test_ece_perfect_calibration(self):
        # All predictions match targets exactly, ECE should be 0
        preds = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
            ]
        )
        ece = compute_calibration_error(preds, target, n_bins=2, norm="l1")
        print("ECE for perfectly calibrated predictions: ", ece)
        bin_confs, bin_accs, bin_weights, ece, mce = _bin_stats(
            preds, target, n_bins=2, quantile=False
        )
        print("ECE from _bin_stats: ", ece)
        assert torch.isclose(ece, torch.tensor(0.0), atol=1e-6)

    def test_ece_worst_calibration(self):
        # All predictions are the opposite of targets, ECE should be 1
        preds = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        target = 1 - preds.int()
        ece = compute_calibration_error(preds, target, n_bins=2, norm="l1")
        print("ECE for worst calibrated predictions: ", ece)
        bin_confs, bin_accs, bin_weights, ece, mce = _bin_stats(
            preds, target, n_bins=2, quantile=False
        )
        print("ECE from _bin_stats for worst calibration: ", ece)
        assert torch.isclose(ece, torch.tensor(1.0), atol=1e-6)
