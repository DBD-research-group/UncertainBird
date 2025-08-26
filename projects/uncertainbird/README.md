# UncertainBird Metrics

This module implements uncertainty quantification metrics for multilabel bird sound classification, specifically focusing on calibration error metrics following the One-vs-All (OvA) decomposition approach.

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest tests/test_metrics.py -v

# Run specific test class  
poetry run python -m pytest tests/test_metrics.py::TestMultilabelECETopK -v

# Run with output for paper example
poetry run python -m pytest tests/test_metrics.py::TestMultilabelECETopK::test_paper_example -v -s

# Run smoke test only
poetry run python tests/test_metrics.py
```

The metrics include comprehensive test coverage with **24 test cases** covering basic functionality, edge cases, device compatibility, realistic scenarios, and paper example validation.

## Metrics Implemented

### 1. MultilabelCalibrationError (ECE_OvA)
Implementation of calibration error for multilabel classification using One-vs-All (OvA) decomposition. Each label is treated as an independent binary classification problem.

**Mathematical Definition:**
```
ECE = Σ_j E[|P[Y_j|φ_j(X) = η_j] - η_j|]
```

### 2. MultilabelECEMarginal (EVEC_Marginal)
Marginal Expected Calibration Error that treats each label independently following the OvA decomposition approach from the paper.

**Usage in evaluation:**
```python
"EVEC_Marginal": MultilabelECEMarginal(num_labels=num_labels, n_bins=10)
```

### 3. MultilabelECETopK (ECE@k)
Top-k Expected Calibration Error that focuses only on the top-k predicted labels, which is more relevant for practical multilabel classification scenarios.

**Mathematical Definition:**
```
ECE@k = E[∑_{j∈top_k φ(X)} |P[Y_j = 1|φ_j(X)] - φ_j(X)|]
```

**Key Features:**
- Addresses practical multilabel prediction where only top-k labels are predicted as relevant
- More meaningful for real-world applications where decisions are made on top-k scoring labels
- As described in the paper: evaluates calibration quality for the labels that matter most in practice

**Usage in evaluation:**
```python
"ECE@5": MultilabelECETopK(num_labels=num_labels, k=5, n_bins=10)
```

## Integration with BirdSet

The metrics are integrated into the BirdSet framework through the `MultilabelMetricsConfig` class:

```python
eval_complete: MetricCollection = MetricCollection({
    "cmAP5": cmAP5(num_labels=num_labels, sample_threshold=5, thresholds=None),
    "pcmAP": pcmAP(num_labels=num_labels, padding_factor=5, average="macro", thresholds=None),
    "ECE_OvA": MultilabelCalibrationError(num_labels=num_labels, n_bins=10),
    "EVEC_Marginal": MultilabelECEMarginal(num_labels=num_labels, n_bins=10),
})
```

## Testing

## Implementation Details

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
