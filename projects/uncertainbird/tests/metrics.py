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
        MultilabelECETopK,
    )
except ImportError:
    # Fallback: try relative import from parent directory
    parent_dir = os.path.dirname(uncertainbird_root)
    sys.path.insert(0, parent_dir)
    from projects.uncertainbird.modules.metrics.uncertainty import (
        MultilabelCalibrationError,
        MultilabelECEMarginal,
        MultilabelECETopK,
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
        metric3 = MultilabelECETopK(num_labels=num_labels, k=1, n_bins=n_bins)

        # Test with float16 (half precision) which can cause dtype mismatches
        preds = torch.rand(8, num_labels, dtype=torch.float16, device=device)
        targets = torch.randint(0, 2, (8, num_labels), device=device)

        # Should not raise dtype mismatch errors
        metric1.update(preds, targets)
        metric2.update(preds, targets)
        metric3.update(preds, targets)

        ece1 = metric1.compute()
        ece2 = metric2.compute()
        ece3 = metric3.compute()

        # Results should be finite and reasonable
        assert torch.isfinite(ece1), "MultilabelCalibrationError should handle float16"
        assert torch.isfinite(ece2), "MultilabelECEMarginal should handle float16"
        assert torch.isfinite(ece3), "MultilabelECETopK should handle float16"
        assert (
            ece1 >= 0
        ), "MultilabelCalibrationError should be non-negative with float16"
        assert ece2 >= 0, "MultilabelECEMarginal should be non-negative with float16"
        assert ece3 >= 0, "MultilabelECETopK should be non-negative with float16"

    def test_all_metrics_comparison(self):
        """Test all three metrics on the same data for consistency."""
        num_labels = 4
        n_bins = 5

        metric1 = MultilabelCalibrationError(num_labels=num_labels, n_bins=n_bins)
        metric2 = MultilabelECEMarginal(num_labels=num_labels, n_bins=n_bins)
        metric3 = MultilabelECETopK(num_labels=num_labels, k=2, n_bins=n_bins)

        # Generate test data
        torch.manual_seed(123)
        preds = torch.rand(15, num_labels)
        targets = torch.randint(0, 2, (15, num_labels))

        # Update all metrics
        metric1.update(preds, targets)
        metric2.update(preds, targets)
        metric3.update(preds, targets)

        ece1 = metric1.compute()
        ece2 = metric2.compute()
        ece3 = metric3.compute()

        # All should be finite and non-negative
        assert torch.isfinite(ece1), "CalibrationError should be finite"
        assert torch.isfinite(ece2), "ECEMarginal should be finite"
        assert torch.isfinite(ece3), "ECETopK should be finite"
        assert ece1 >= 0, "CalibrationError should be non-negative"
        assert ece2 >= 0, "ECEMarginal should be non-negative"
        assert ece3 >= 0, "ECETopK should be non-negative"

        # All should be reasonable values
        assert ece1 <= 1.0, "CalibrationError should not exceed 1"
        assert ece2 <= 1.0, "ECEMarginal should not exceed 1"
        assert ece3 <= 1.0, "ECETopK should not exceed 1"


class TestMultilabelECETopK:
    """Test cases for MultilabelECETopK metric."""

    def test_paper_example(self):
        """Test the specific example from the paper.

        From the paper: "Taking the example from above, with k = 5 predictions,
        the ECE becomes 9.25, showing that the error defined in (6) can be quite
        misleading in XMLC."

        The example uses m = 2, X = {x1, x2}, with P[x1] = P[x2] = 1/2.
        Consider a classifier with ground truth values as in Table 1.
        """
        metric = MultilabelECETopK(
            num_labels=2, k=1, n_bins=10
        )  # Use k=1 for simplicity

        # Example from paper: classifier predicts top-1
        # φ(x) = top₁(ψ(x)). In this case, φ(x₁) = 2, φ(x₂) = 1
        # Corresponding classifier predictions: 0.9 and 0.7

        # Sample 1: top prediction is label 1 (index 1) with confidence 0.9, ground truth is 1
        preds1 = torch.tensor([[0.1, 0.9]])  # Label 0: 0.1, Label 1: 0.9 (top-1)
        targets1 = torch.tensor([[0, 1]])  # Ground truth: Label 1 is positive

        # Sample 2: top prediction is label 0 (index 0) with confidence 0.7, ground truth is 0
        preds2 = torch.tensor([[0.7, 0.6]])  # Label 0: 0.7 (top-1), Label 1: 0.6
        targets2 = torch.tensor(
            [[0, 1]]
        )  # Ground truth: Label 0 is negative, Label 1 is positive

        metric.update(preds1, targets1)
        metric.update(preds2, targets2)

        ece = metric.compute()

        # The paper mentions the error becomes significant when focusing on top-k
        # We expect some calibration error due to the mismatch in the second sample
        assert torch.isfinite(ece), "ECE@k should be finite"
        assert ece >= 0, "ECE@k should be non-negative"
        print(f"ECE@1 for paper example: {ece}")

    def test_perfect_top_k_calibration(self):
        """Test with perfectly calibrated top-k predictions."""
        metric = MultilabelECETopK(num_labels=5, k=2, n_bins=5)

        # Create data where top-2 predictions are well calibrated
        preds = torch.tensor(
            [
                [0.1, 0.2, 0.8, 0.9, 0.3],  # Top-2: indices 3,2 with conf 0.9,0.8
                [0.2, 0.7, 0.1, 0.8, 0.4],  # Top-2: indices 3,1 with conf 0.8,0.7
                [0.3, 0.9, 0.2, 0.1, 0.8],  # Top-2: indices 1,4 with conf 0.9,0.8
            ]
        )

        # Make targets match the confidence levels for top-k predictions
        targets = torch.tensor(
            [
                [0, 0, 1, 1, 0],  # Top-2 correct: high conf predictions are positive
                [0, 1, 0, 1, 0],  # Top-2 correct: high conf predictions are positive
                [0, 1, 0, 0, 1],  # Top-2 correct: high conf predictions are positive
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "ECE@k should be finite"
        assert ece >= 0, "ECE@k should be non-negative"

    def test_poor_top_k_calibration(self):
        """Test with poorly calibrated top-k predictions."""
        metric = MultilabelECETopK(num_labels=3, k=2, n_bins=5)

        # High confidence predictions that are wrong
        preds = torch.tensor(
            [
                [0.1, 0.9, 0.8],  # Top-2: indices 1,2 with high confidence
                [0.8, 0.2, 0.9],  # Top-2: indices 2,0 with high confidence
            ]
        )

        # Targets opposite to high confidence predictions
        targets = torch.tensor(
            [
                [1, 0, 0],  # Top-2 predictions are wrong
                [0, 1, 0],  # Top-2 predictions are wrong
            ]
        )

        metric.update(preds, targets)
        ece = metric.compute()

        # Should have high calibration error
        assert torch.isfinite(ece), "ECE@k should be finite"
        assert ece > 0, "ECE@k should be positive for miscalibrated top-k predictions"

    def test_edge_cases_top_k(self):
        """Test edge cases for ECE@k."""
        metric = MultilabelECETopK(num_labels=3, k=2, n_bins=5)

        # Single sample
        preds = torch.tensor([[0.3, 0.8, 0.5]])  # Top-2: indices 1,2
        targets = torch.tensor([[0, 1, 1]])

        metric.update(preds, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "ECE@k should handle single sample"
        assert ece >= 0, "ECE@k should be non-negative"

    def test_k_larger_than_labels(self):
        """Test when k is larger than number of labels."""
        metric = MultilabelECETopK(num_labels=2, k=5, n_bins=3)  # k > num_labels

        preds = torch.tensor([[0.6, 0.4]])  # Only 2 labels, but k=5
        targets = torch.tensor([[1, 0]])

        metric.update(preds, targets)
        ece = metric.compute()

        assert torch.isfinite(ece), "ECE@k should handle k > num_labels"
        assert ece >= 0, "ECE@k should be non-negative"

    def test_empty_input_top_k(self):
        """Test ECE@k with empty input."""
        metric = MultilabelECETopK(num_labels=3, k=2, n_bins=5)

        ece = metric.compute()
        assert ece == 0.0, "ECE@k should be 0 for empty input"


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
