<div align="center">
  <img src="https://github.com/DBD-research-group/BirdSet/blob/main/resources/perch/birdsetsymbol.png" alt="logo" width="100">
</div>

# UncertainBird -  🤗
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-ffcc00?logo=huggingface&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/DBD-research-group/BirdSet"><img alt="GitHub: github.com/DBD-research-group/BirdSet " src="https://img.shields.io/badge/-BirdSet-017F2F?style=flat&logo=github&labelColor=gray"></a>
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2403.10380) -->

# 


This project addresses the challenge of uncertainty estimation in AI-drivenbird sound classification, essential for the ”DeepBirdDetect” project aimed at harmonizing wind power expansion with avian conservation. We aim to evaluate methods such as Monte Carlo Dropout, Spectral-normalized Neural GaussianProcess, and Focal Loss within deep learning frameworks, assessing their performance across various neural network architectures, including CNNs and Trans-formers, and model scales. Our findings will provide insights into the suitability of these uncertainty estimation techniques for environmental conservation applications, offering a basis for more reliable and transparent AI-based wildlife monitoring.

## User Installation

The simplest way to install $\texttt{UncertainBird}$ is to clone this repository.

You can also use the [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) configured as as git submodule:
```bash
git submodule update --init --recursive
```

And install python dependencies with [uv](https://docs.astral.sh/uv/).
```
uv sync
```

Install TF specific dependencies:
```bash
uv pip install -r projects/UncertainBird/tf-requirements.txt
```
Activate virtual environment:
```
eval ./venv/bin/activate
```

## Dump predictions for calibration analysis

```bash
./projects/UncertainBird/scripts/dump-predictions.sh --models <model1,model2,...> --dataset <dataset_name> --batch_size <batch_size> --gpus <gpu_id>
```

Dump perch v2 predictions for models trained on BirdSet:
```bash
python ./projects/uncertainbird/scripts/dump_perch_predictions.py --datasets HSN --gpu 2 --output-dir /workspace/logs/predictions/perch_v2
```

## Run experiments

Our experiments are defined in the `projects/UncertainBird/configs/experiment/` folder. To run an experiment, use the following command in the directory of the repository:

``` bash
./projects/UncertainBird/train.sh experiment="EXPERIMENT_PATH"
```

E.g.
``` bash
./projects/uncertainbird/train.sh experiment=convnext/masked   
```

## Project structure

This repository is a fork of [BirdSet](https://github.com/DBD-research-group/BirdSet). All project related changes are made in the `projects/UncertainBird` folder. If you want to change or fixed a bug in the original code, please make a pull request to the original repository. You can use all configurations and scripts from the original repository. If you want to override the configurations add a file with the appropriate path in the `projects/UncertainBird/configs/` folder. Python code can be added in the `projects/UncertainBird/src` folder use the same folder structure as in `/birdset`.

## Experiments

### Compare calibration error with and without masking

- ./projects/uncertainbird/train.sh experiment=convnext/masked  
- ./projects/uncertainbird/train.sh experiment=convnext/base 


## Implemented Uncertainty Metrics

This project implements advanced calibration error metrics for multilabel bird sound classification:

### Calibration Error Metrics Suite

Four complementary calibration error metrics for multilabel classification, following the One-vs-All (OvA) decomposition approach:

- **MultilabelCalibrationError**: Standard Expected Calibration Error (ECE) with uniform binning. Supports 'marginal', 'weighted', and 'global' averaging across classes.
- **TopKMultiLabelCalibrationError**: Computes ECE for only the top-K most relevant classes, with flexible selection criteria ('probability', 'predicted class', or 'target class').


**Key Features:**
- Robust NaN handling for production environments
- CUDA and mixed precision support for modern training pipelines
- Comprehensive test coverage with 26 test cases across all metric classes
- Device/dtype compatibility for distributed training
- Refactored codebase with utility functions to eliminate code duplication

**Quick Test:**
```bash
cd projects/uncertainbird
poetry run pytest tests/test_metrics.py -v  # Full test suite (26 tests)
poetry run python tests/test_metrics.py    # Smoke test
```

For detailed documentation, see [projects/uncertainbird/README.md](projects/uncertainbird/README.md).




