# UncertainBird Metrics

This module implements uncertainty quantification metrics for multilabel bird sound classification, specifically focusing on calibration error metrics following the One-vs-All (OvA) decomposition approach.

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest tests/test_metrics.py -v

# Run specific test class  
poetry run python -m pytest tests/test_metrics.py::TestMultilabelCalibrationError -v
poetry run python -m pytest tests/test_metrics.py::TestMultilabelECEMarginal -v
poetry run python -m pytest tests/test_metrics.py::TestMultilabelECETopK -v
poetry run python -m pytest tests/test_metrics.py::TestMetricComparison -v

# Run with output for specific tests
poetry run python -m pytest tests/test_metrics.py::TestMultilabelECETopK::test_ece_topk_equals_ece_ova_when_k_equals_num_labels -v -s

# Run smoke test only
poetry run python tests/test_metrics.py
```

The metrics include comprehensive test coverage with **26 test cases** across **4 test classes** covering:
- Basic functionality and mathematical correctness
- Edge cases and error handling
- Device/dtype compatibility for distributed training
- Realistic batch scenarios and paper example validation
- Cross-metric comparisons (ECE@k equals ECE_OvA when k=num_labels)

## Metrics Implemented

### 1. MultilabelCalibrationError (ECE_OvA)
Implementation of calibration error for multilabel classification using One-vs-All (OvA) decomposition. Each label is treated as an independent binary classification problem.

**Mathematical Definition:**
```
ECE = Σ_j E[|P[Y_j|φ_j(X) = η_j] - η_j|]
```

**Key Features:**
- Standard Expected Calibration Error with uniform binning
- Robust NaN handling and edge case management
- Compatible with distributed training and mixed precision

**Usage in evaluation:**
```python
"ECE_OvA": MultilabelCalibrationError(num_labels=num_labels, n_bins=10)
```

### 2. MultilabelECEMarginal (ECE_Marginal)
Marginal Expected Calibration Error that treats each label independently following the OvA decomposition approach from the paper.

**Key Features:**
- Accumulates statistics separately for each label
- Computes ECE per label then averages across labels
- More granular analysis of per-label calibration quality

**Usage in evaluation:**
```python
"ECE_Marginal": MultilabelECEMarginal(num_labels=num_labels, n_bins=10)
```

### 3. MultilabelECETopK (ECE@k)
Top-k Expected Calibration Error that focuses only on the top-k most frequently predicted classes across the dataset. This is more relevant for practical multilabel classification scenarios.

**Mathematical Definition:**
```
ECE@k = E[∑_{j∈top_k(φ(X))} |P[Y_j = 1|φ_j(X)] - φ_j(X)|]
```

**Key Features:**
- Identifies top-k classes by highest maximum prediction scores
- Evaluates calibration only for the most relevant/active labels
- Addresses practical multilabel prediction where only top-k labels matter
- More meaningful for real-world applications where decisions are made on top-k scoring labels

**Usage in evaluation:**
```python
"ECE@5": MultilabelECETopK(num_labels=num_labels, k=5, n_bins=10)
"ECE@10": MultilabelECETopK(num_labels=num_labels, k=10, n_bins=10)
"ECE@num_labels": MultilabelECETopK(num_labels=num_labels, k=num_labels, n_bins=10)
```

### 4. MultilabelACE (ACE_OvA)
Adaptive Calibration Error using quantile-based binning where each bin contains an equal number of samples, different from uniform binning used in ECE.

**Mathematical Definition:**
```
ACE = Σ_j E[|P[Y_j|φ_j(X) ∈ B_q] - B̄_q|]
```
where B_q are quantile-based bins with equal sample counts.

**Key Features:**
- Quantile-based binning for more adaptive calibration assessment
- Each bin contains approximately equal number of predictions
- More robust when predictions are not uniformly distributed
- Supports multiple norms: L1, L2, and max norm

**Usage in evaluation:**
```python
"ACE_OvA": MultilabelACE(num_labels=num_labels, n_bins=10, norm="l1")
```

## Integration with BirdSet

The metrics are integrated into the BirdSet framework through the `MultilabelMetricsConfig` class:

```python
eval_complete: MetricCollection = MetricCollection({
    "cmAP5": cmAP5(num_labels=num_labels, sample_threshold=5, thresholds=None),
    "pcmAP": pcmAP(num_labels=num_labels, padding_factor=5, average="macro", thresholds=None),
    "ECE_OvA": MultilabelCalibrationError(num_labels=num_labels, n_bins=10),
    "ECE_Marginal": MultilabelECEMarginal(num_labels=num_labels, n_bins=10),
    "ECE@5": MultilabelECETopK(num_labels=num_labels, k=5, n_bins=10),
    "ECE@10": MultilabelECETopK(num_labels=num_labels, k=10, n_bins=10), 
    "ECE@num_labels": MultilabelECETopK(num_labels=num_labels, k=num_labels, n_bins=10),
    "ACE_OvA": MultilabelACE(num_labels=num_labels, n_bins=10),
})
```

**Expected Output Metrics:**
- `test/ECE_OvA`: Standard One-vs-All calibration error
- `test/ECE_Marginal`: Marginal calibration error with per-label averaging  
- `test/ECE@5`: Top-5 calibration error for practical prediction scenarios
- `test/ECE@10`: Top-10 calibration error
- `test/ECE@num_labels`: Full calibration error (equivalent to ECE_OvA when k=num_labels)
- `test/ACE_OvA`: Adaptive calibration error with quantile-based binning

## Testing

## Implementation Details

### Code Architecture & Refactoring

The implementation has been refactored to eliminate code duplication with dedicated utility functions:

- **`_sigmoid_and_clamp()`**: Consistent logit-to-probability conversion and clamping
- **`_create_uniform_bins()`** & **`_assign_to_bins()`**: Uniform binning utilities
- **`_process_bins_for_calibration()`**: Uniform binning for ECE metrics
- **`_process_quantile_bins_for_ace()`**: Quantile-based binning for ACE metric

This refactoring reduced code duplication by ~200 lines while maintaining full backward compatibility and test coverage.

### Key Features Implemented

1. **NaN Prevention**: Robust handling of edge cases that could cause NaN values
2. **Device Management**: Automatic device handling following PyTorch Lightning best practices
3. **Dtype Consistency**: Proper handling of mixed precision and different tensor types
4. **Memory Efficiency**: Optimized for large-scale training with multiple labels
5. **Distributed Training**: Compatible with multi-GPU setups

### Performance Considerations

- **Binning Strategy**: Manual binning implementation for better control and robustness
- **State Management**: Efficient accumulation of statistics across training batches
- **Error Handling**: Graceful fallback mechanisms for edge cases

## References

The implementation follows the One-vs-All (OvA) decomposition approach for multilabel calibration as described in:

**"Calibration Error for Multi-Class and Multi-Label Prediction"** ([arXiv:2411.04276](https://arxiv.org/abs/2411.04276))

The metrics implement the paper's approach where each label is treated as an independent binary classification task for calibration purposes, enabling proper uncertainty quantification in multilabel bird sound classification.
