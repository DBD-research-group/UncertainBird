<div align="center">
  <img src="https://github.com/DBD-research-group/BirdSet/blob/main/resources/perch/birdsetsymbol.png" alt="logo" width="100">
</div>

# Uncertainty Calibration of Multi-Label Bird Sound Classifiers
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-ffcc00?logo=huggingface&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/DBD-research-group/BirdSet"><img alt="GitHub: github.com/DBD-research-group/BirdSet " src="https://img.shields.io/badge/-BirdSet-017F2F?style=flat&logo=github&labelColor=gray"></a>
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2403.10380) -->

## Abstract

  Passive acoustic monitoring enables large-scale biodiversity assessment, but reliable classification of bioacoustic sounds requires not only high accuracy but also well-calibrated uncertainty estimates to ground decision-making. In bioacoustics, calibration is challenged by overlapping vocalisations, long-tailed species distributions, and distribution shifts between training and deployment data. The calibration of multi-label deep learning classifiers within the domain of bioacoustics has not yet been assessed. We systematically benchmark the calibration of four state-of-the-art multi-label bird sound classifiers on the BirdSet benchmark, evaluating global, per-dataset, and per-class calibration using threshold-free calibration metrics (ECE, MCS) alongside discrimination metrics (cmAP).
  Model calibration varies significantly across datasets and classes. While Perch v2 and ConvNeXt$_{BS}$ show better global calibration, results vary between datasets. Both models indicate consistent underconfidence, while AudioProtoPNet and BirdMAE are mostly overconfident.
  Surprisingly, calibration seems to be better for less frequent classes. Using simple post hoc calibration methods we demonstrate a straightforward way to improve calibration. A small labelled calibration set is sufficient to significantly improve calibration with Platt scaling, while global calibration parameters suffer from dataset variability. Our findings highlight the importance of evaluating and improving uncertainty calibration in bioacoustic classifiers.

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
python ./uncertainbird/scripts/dump_predictions.py  --model <model> --dataset <dataset_names> --gpu <gpu_id> --output-dir <output_dir> --num-workers <num_workers> 
```

For example:
```bash
python ./uncertainbird/scripts/dump_predictions.py  --model convnext_bs --datasets NBP HSN --gpu 0 --output-dir ./logs/predictions --num-workers 1
```

## Experiments

### Benchmarking the Calibration of Bird sound Classifiers

See notebooks in `uncertainbird/benchmarking_calibration` for details. [This notebook](uncertainbird/benchmarking_calibration/Calibration_Benchmarking.ipynb) provides an overview of the experiments and results in the paper.

### Platt & Temperature Scaling

See notebooks in `uncertainbird/notebooks/posthoc_calibration` for details.



## Citation

```bib
@misc{schwinger2025uncertaintycalibrationmultilabelbird,
      title={Uncertainty Calibration of Multi-Label Bird Sound Classifiers}, 
      author={Raphael Schwinger and Ben McEwen and Vincent S. Kather and René Heinrich and Lukas Rauch and Sven Tomforde},
      year={2025},
      eprint={2511.08261},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2511.08261}, 
}
```