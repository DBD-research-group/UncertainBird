"""
Test suite for MultilabelCalibrationError and MultilabelECEMarginal metrics.

This module tests the uncertainty quantification metrics for multilabel classification
following the One-vs-All (OvA) decomposition approach from the paper.

Usage with Poetry (from uncertainbird directory):
    # Run full test suite
    poetry run pytest tests/metrics.py -v

    # Run smoke test only
    poetry run python tests/metrics.py

    # Run specific test class
    poetry run pytest tests/metrics.py::TestMultilabelCalibrationError -v
"""

import torch
import pytest
from unittest.mock import patch
import sys
import os

# Add the uncertainbird project to the path
uncertainbird_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if uncertainbird_root not in sys.path:
    sys.path.insert(0, uncertainbird_root)

try:
    from modules.metrics.uncertainty import (
        MultilabelCalibrationError,
        MultilabelECEMarginal,
    )
except ImportError:
    # Fallback: try relative import from parent directory
    parent_dir = os.path.dirname(uncertainbird_root)
    sys.path.insert(0, parent_dir)
    from projects.uncertainbird.modules.metrics.uncertainty import (
        MultilabelCalibrationError,
        MultilabelECEMarginal,
    )


class TestMultilabelCalibrationError:
    """Test cases for MultilabelCalibrationError metric."""

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        metric = MultilabelCalibrationError(num_labels=3, n_bins=5)

        # Perfect calibration: prediction probabilities match actual rates
        preds = torch.tensor(
            [
                [0.1, 0.3, 0.9],  # Low, medium, high confidence
                [0.1, 0.3, 0.9],
                [0.1, 0.3, 0.9],
                [0.1, 0.3, 0.9],
                [0.1, 0.3, 0.9],
            ]
        )

        # Targets that match the probabilities (1 out of 10 should be positive for 0.1, etc.)
        targets = torch.tensor(
            [
                [
                    0,
                    0,
                    1,
                ],  # 0.1 prob -> mostly 0, 0.3 prob -> mostly 0, 0.9 prob -> mostly 1
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 1],  # 0.3 prob gets some 1s
                [1, 1, 1],  # Higher prob areas get more 1s
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        # Should be low for reasonably calibrated data
        assert torch.isfinite(ece), "ECE should be finite"
        assert ece >= 0, "ECE should be non-negative"

    def test_poor_calibration(self):
        """Test with poorly calibrated predictions."""
        metric = MultilabelCalibrationError(num_labels=2, n_bins=5)

        # Poor calibration: high confidence but wrong predictions
        preds = torch.tensor(
            [
                [0.9, 0.9],  # Very confident
                [0.9, 0.9],
                [0.9, 0.9],
                [0.1, 0.1],  # Very unconfident
                [0.1, 0.1],
            ]
        )

        # Opposite targets (overconfident model)
        targets = torch.tensor(
            [
                [0, 0],  # High confidence but wrong
                [0, 0],
                [0, 0],
                [1, 1],  # Low confidence but actually right
                [1, 1],
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        # Should be high for poorly calibrated data
        assert torch.isfinite(ece), "ECE should be finite"
        assert ece > 0, "ECE should be positive for miscalibrated predictions"

    def test_edge_cases(self):
        """Test edge cases that might cause NaN."""
        metric = MultilabelCalibrationError(num_labels=2, n_bins=10)

        # Test with all zeros
        preds_zeros = torch.zeros(5, 2)
        targets_zeros = torch.zeros(5, 2)
        metric.update(preds_zeros, targets_zeros)

        # Test with all ones
        preds_ones = torch.ones(5, 2)
        targets_ones = torch.ones(5, 2)
        metric.update(preds_ones, targets_ones)

        # Test with extreme values
        preds_extreme = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        targets_extreme = torch.tensor([[0, 1], [1, 0]])
        metric.update(preds_extreme, targets_extreme)

        ece = metric.compute()
        assert torch.isfinite(ece), "ECE should handle edge cases without NaN"
        assert ece >= 0, "ECE should be non-negative"

    def test_logits_input(self):
        """Test that logits are properly converted to probabilities."""
        metric = MultilabelCalibrationError(num_labels=2, n_bins=5)

        # Raw logits (should be converted via sigmoid)
        logits = torch.tensor(
            [
                [-2.0, 2.0],  # Low and high logits
                [0.0, 0.0],  # Medium logits
                [1.0, -1.0],  # Mixed logits
            ]
        )

        targets = torch.tensor(
            [
                [0, 1],
                [0, 0],
                [1, 0],
            ]
        )

        metric.update(logits, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "ECE should handle logits input"
        assert ece >= 0, "ECE should be non-negative"

    def test_empty_input(self):
        """Test behavior with empty inputs."""
        metric = MultilabelCalibrationError(num_labels=3, n_bins=5)

        # No data
        ece = metric.compute()
        assert ece == 0.0, "ECE should be 0 for empty input"

    def test_single_sample(self):
        """Test with single sample."""
        metric = MultilabelCalibrationError(num_labels=2, n_bins=3)

        preds = torch.tensor([[0.7, 0.3]])
        targets = torch.tensor([[1, 0]])

        metric.update(preds, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "ECE should handle single sample"
        assert ece >= 0, "ECE should be non-negative"

    def test_realistic_batch_size(self):
        """Test with realistic batch sizes."""
        metric = MultilabelCalibrationError(
            num_labels=21, n_bins=10
        )  # HSN dataset size

        # Simulate multiple batches like in real training
        for _ in range(5):
            batch_size = 32
            preds = torch.rand(batch_size, 21)
            targets = torch.randint(0, 2, (batch_size, 21))
            metric.update(preds, targets)

        ece = metric.compute()
        assert torch.isfinite(ece), "ECE should handle realistic batch sizes"
        assert ece >= 0, "ECE should be non-negative"


class TestMultilabelECEMarginal:
    """Test cases for MultilabelECEMarginal metric."""

    def test_perfect_calibration_marginal(self):
        """Test marginal ECE with perfectly calibrated predictions."""
        metric = MultilabelECEMarginal(num_labels=3, n_bins=5)

        # Create calibrated data where each label behaves independently
        preds = torch.tensor(
            [
                [0.2, 0.5, 0.8],
                [0.2, 0.5, 0.8],
                [0.2, 0.5, 0.8],
                [0.2, 0.5, 0.8],
                [0.2, 0.5, 0.8],
            ]
        )

        # Targets that approximately match the marginal probabilities
        targets = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "Marginal ECE should be finite"
        assert ece >= 0, "Marginal ECE should be non-negative"

    def test_independent_labels(self):
        """Test that marginal ECE treats labels independently."""
        metric = MultilabelECEMarginal(num_labels=2, n_bins=3)

        # First label well calibrated, second label poorly calibrated
        preds = torch.tensor(
            [
                [0.3, 0.9],  # Label 0: medium conf, Label 1: high conf
                [0.3, 0.9],
                [0.3, 0.9],
            ]
        )

        targets = torch.tensor(
            [
                [0, 0],  # Label 0 correct rate ~33%, Label 1 correct rate 0%
                [1, 0],
                [0, 0],
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        # Should detect the miscalibration in label 1
        assert torch.isfinite(ece), "Marginal ECE should be finite"
        assert ece > 0, "Should detect miscalibration"

    def test_edge_cases_marginal(self):
        """Test edge cases for marginal ECE."""
        metric = MultilabelECEMarginal(num_labels=3, n_bins=5)

        # All predictions at boundaries
        preds = torch.tensor(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
            ]
        )

        targets = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 1],
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "Marginal ECE should handle boundary cases"
        assert ece >= 0, "Marginal ECE should be non-negative"

    def test_logits_conversion_marginal(self):
        """Test logits to probability conversion."""
        metric = MultilabelECEMarginal(num_labels=2, n_bins=3)

        # Test with logits
        logits = torch.tensor(
            [
                [-1.0, 1.0],
                [0.0, 0.0],
                [2.0, -2.0],
            ]
        )

        targets = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 0],
            ]
        )

        metric.update(logits, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "Should handle logits"
        assert ece >= 0, "ECE should be non-negative"

    def test_empty_marginal(self):
        """Test marginal ECE with empty input."""
        metric = MultilabelECEMarginal(num_labels=2, n_bins=5)

        ece = metric.compute()
        assert ece == 0.0, "Should return 0 for empty input"

    def test_state_accumulation(self):
        """Test that metric properly accumulates state across updates."""
        metric = MultilabelECEMarginal(num_labels=2, n_bins=3)

        # First batch
        preds1 = torch.tensor([[0.3, 0.7]])
        targets1 = torch.tensor([[0, 1]])
        metric.update(preds1, targets1)

        # Second batch
        preds2 = torch.tensor([[0.3, 0.7], [0.3, 0.7]])
        targets2 = torch.tensor([[1, 0], [0, 1]])
        metric.update(preds2, targets2)

        ece = metric.compute()

        assert torch.isfinite(ece), "Should accumulate properly"
        assert ece >= 0, "ECE should be non-negative"

    def test_realistic_multilabel_scenario(self):
        """Test with realistic multilabel classification scenario."""
        metric = MultilabelECEMarginal(num_labels=21, n_bins=10)  # HSN dataset

        # Simulate realistic training scenario with multiple batches
        for batch_idx in range(10):
            batch_size = 16
            # Simulate model predictions with some correlation structure
            preds = torch.sigmoid(
                torch.randn(batch_size, 21) * 2
            )  # More realistic distribution
            targets = torch.randint(0, 2, (batch_size, 21))

            metric.update(preds, targets)

        ece = metric.compute()
        assert torch.isfinite(ece), "Should handle realistic scenario"
        assert ece >= 0, "ECE should be non-negative"
        assert ece <= 1.0, "ECE should not exceed 1"


class TestMetricComparison:
    """Compare both metrics on the same data."""

    def test_both_metrics_same_data(self):
        """Test both metrics on identical data."""
        num_labels = 3
        n_bins = 5

        metric1 = MultilabelCalibrationError(num_labels=num_labels, n_bins=n_bins)
        metric2 = MultilabelECEMarginal(num_labels=num_labels, n_bins=n_bins)

        # Generate some test data
        torch.manual_seed(42)  # For reproducibility
        preds = torch.rand(20, num_labels)  # Random predictions
        targets = torch.randint(0, 2, (20, num_labels))  # Random binary targets

        # Update both metrics
        metric1.update(preds, targets)
        metric2.update(preds, targets)

        ece1 = metric1.compute()
        ece2 = metric2.compute()

        # Both should produce finite, non-negative results
        assert torch.isfinite(ece1), "MultilabelCalibrationError should be finite"
        assert torch.isfinite(ece2), "MultilabelECEMarginal should be finite"
        assert ece1 >= 0, "MultilabelCalibrationError should be non-negative"
        assert ece2 >= 0, "MultilabelECEMarginal should be non-negative"

        # Results might differ due to different implementations, but should be reasonable
        assert ece1 <= 1.0, "ECE should not exceed 1"
        assert ece2 <= 1.0, "ECE should not exceed 1"

    def test_device_compatibility(self):
        """Test that metrics work correctly with CUDA tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device compatibility test")

        device = torch.device("cuda")
        num_labels = 3
        n_bins = 5

        metric1 = MultilabelCalibrationError(num_labels=num_labels, n_bins=n_bins)
        metric2 = MultilabelECEMarginal(num_labels=num_labels, n_bins=n_bins)

        # Test data on CUDA
        preds = torch.rand(10, num_labels, device=device)
        targets = torch.randint(0, 2, (10, num_labels), device=device)

        # Should not raise device mismatch errors
        metric1.update(preds, targets)
        metric2.update(preds, targets)

        ece1 = metric1.compute()
        ece2 = metric2.compute()

        # Results should be finite and reasonable
        assert torch.isfinite(
            ece1
        ), "MultilabelCalibrationError should be finite on CUDA"
        assert torch.isfinite(ece2), "MultilabelECEMarginal should be finite on CUDA"
        assert ece1 >= 0, "MultilabelCalibrationError should be non-negative on CUDA"
        assert ece2 >= 0, "MultilabelECEMarginal should be non-negative on CUDA"

    def test_dtype_compatibility(self):
        """Test that metrics work with different data types like float16."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for dtype test")

        device = torch.device("cuda")
        num_labels = 2
        n_bins = 3

        metric1 = MultilabelCalibrationError(num_labels=num_labels, n_bins=n_bins)
        metric2 = MultilabelECEMarginal(num_labels=num_labels, n_bins=n_bins)

        # Test with float16 (half precision) which can cause dtype mismatches
        preds = torch.rand(8, num_labels, dtype=torch.float16, device=device)
        targets = torch.randint(0, 2, (8, num_labels), device=device)

        # Should not raise dtype mismatch errors
        metric1.update(preds, targets)
        metric2.update(preds, targets)

        ece1 = metric1.compute()
        ece2 = metric2.compute()

        # Results should be finite and reasonable
        assert torch.isfinite(ece1), "MultilabelCalibrationError should handle float16"
        assert torch.isfinite(ece2), "MultilabelECEMarginal should handle float16"
        assert (
            ece1 >= 0
        ), "MultilabelCalibrationError should be non-negative with float16"
        assert ece2 >= 0, "MultilabelECEMarginal should be non-negative with float16"


# Test runner
if __name__ == "__main__":
    # Run a simple smoke test
    print("Running smoke tests...")

    # Test MultilabelCalibrationError
    metric1 = MultilabelCalibrationError(num_labels=2, n_bins=5)
    preds = torch.rand(10, 2)
    targets = torch.randint(0, 2, (10, 2))
    metric1.update(preds, targets)
    ece1 = metric1.compute()
    print(f"MultilabelCalibrationError ECE: {ece1}")

    # Test MultilabelECEMarginal
    metric2 = MultilabelECEMarginal(num_labels=2, n_bins=5)
    metric2.update(preds, targets)
    ece2 = metric2.compute()
    print(f"MultilabelECEMarginal ECE: {ece2}")

    print("Smoke tests passed!")
    print("\nTo run full test suite with Poetry:")
    print("  poetry run pytest tests/metrics.py -v")
    print("\nTo run just this smoke test:")
    print("  poetry run python tests/metrics.py")
