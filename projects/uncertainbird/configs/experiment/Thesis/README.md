<div align="center">
  <img src="https://github.com/DBD-research-group/BirdSet/blob/main/resources/perch/birdsetsymbol.png" alt="logo" width="100">
</div>

# UncertainBird -  Thesis Mats von Holten
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-ffcc00?logo=huggingface&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/DBD-research-group/BirdSet"><img alt="GitHub: github.com/DBD-research-group/BirdSet " src="https://img.shields.io/badge/-BirdSet-017F2F?style=flat&logo=github&labelColor=gray"></a>
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2403.10380) -->

# 


This Folder is dedicated to the Experiments conducted in the scope of this Thesis, it is split into EAT and ConvNeXT.
The required Hydra configuration files, to reproduce the results are in the folders together with the outputted predictions. Even thoug, we tried to keep our changes locally, we needed to apply some changes to the global birdset repository, so it is imortant, to take exactely this code.

## User Installation

To reproduce the results, run the yaml files with the dumplogic checkpoint active, and plug its outputs into the evaluation notebooks.


## Dump predictions for calibration analysis

```bash
./projects/UncertainBird/scripts/dump-predictions.sh --models <model1,model2,...> --dataset <dataset_name> --batch_size <batch_size> --gpus <gpu_id>
```

## Run experiments

Our experiments are defined in the `projects/UncertainBird/configs/experiment/Thesis/` folder. To run an experiment, use the following command in the directory of the repository:

``` bash
./projects/UncertainBird/train.sh experiment="EXPERIMENT_PATH"
```

E.g.
``` bash
./projects/uncertainbird/train.sh experiment=convnext/masked   
```

