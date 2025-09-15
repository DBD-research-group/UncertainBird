# Example Usage of DumpPredictionsCallback

## Basic Usage

The `DumpPredictionsCallback` can be used to save all predictions and targets from a test run.

### 1. Using in Configuration Files

To use the callback in your experiment configuration, include it in your callbacks section:

```yaml
# In your experiment config file (e.g., masked.yaml)
defaults:
  - override /callbacks: with_predictions.yaml  # This includes dump_predictions
  # ... other defaults
```

Or add it directly:

```yaml
callbacks:
  dump_predictions:
    _target_: projects.uncertainbird.callbacks.DumpPredictionsCallback
    save_dir: "${paths.output_dir}/predictions"
    filename_prefix: "test_predictions"
    save_format: "pickle"
    save_logits: true
```

### 2. Using in Python Code

```python
from projects.uncertainbird.callbacks import DumpPredictionsCallback

# Create the callback
callback = DumpPredictionsCallback(
    save_dir="./predictions",
    filename_prefix="my_test_predictions",
    save_format="pickle",
    save_logits=True
)

# Add to trainer
trainer = pl.Trainer(
    callbacks=[callback],
    # ... other trainer args
)

# Run testing
trainer.test(model, datamodule)
```

### 3. Parameters

- `save_dir`: Directory where predictions will be saved (default: "predictions")
- `filename_prefix`: Prefix for saved files (default: "test_predictions") 
- `save_format`: Format to save data - "pickle" or "numpy" (default: "pickle")
- `save_logits`: Whether to save raw logits (True) or apply output activation (False)

### 4. Output Files

The callback saves:

**Pickle format (recommended):**
- Single `.pkl` file containing:
  - `predictions`: Tensor of all predictions
  - `targets`: Tensor of all targets  
  - `metadata`: Information about the run

**Numpy format:**
- `*_predictions_*.npy`: Predictions as numpy array
- `*_targets_*.npy`: Targets as numpy array
- `*_metadata_*.pkl`: Metadata as pickle file

### 5. Loading Saved Data

```python
import pickle
import torch

# Load pickle format
with open('test_predictions_20250827_143022.pkl', 'rb') as f:
    data = pickle.load(f)
    
predictions = data['predictions']  # torch.Tensor
targets = data['targets']          # torch.Tensor
metadata = data['metadata']        # dict with run information

# Load numpy format
import numpy as np
predictions = np.load('test_predictions_predictions_20250827_143022.npy')
targets = np.load('test_predictions_targets_20250827_143022.npy')
```

This callback is particularly useful for:
- Post-hoc analysis of model predictions
- Computing additional metrics not available during training
- Analyzing prediction confidence and uncertainty
- Creating visualizations and reports
- Comparing different model outputs
