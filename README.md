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

To experiment with Perch_v2 install TF specific dependencies separately:
```bash
uv pip install -r tf-requirements.txt
```
Activate virtual environment:
```
eval ./venv/bin/activate
```

## Dump predictions for calibration analysis

```bash
python ./projects/UncertainBird/scripts/dump-predictions.sh --model <model> --dataset <dataset_names> --gpu <gpu_id> --output-dir <output_dir> --num-workers <num_workers> 
```

For example:
```bash
python ./projects/UncertainBird/scripts/dump-predictions.sh --model convnext_bs --datasets NBP,HSN --gpu 0 --output-dir ./logs/predictions --num-workers 1
```

## Experiments

### Benchmarking the Calibration of Bird sound Classifiers

See notebooks in `uncertainbird/benchmarking_calibration` for details. [This notebook](uncertainbird/benchmarking_calibration/Calibration_Benchmarking.ipynb) provides an overview of the experiments and results in the paper.

### Platt & Temperature Scaling

See notebooks in `uncertainbird/posthoc_calibration` for details.



