#!/bin/bash

# Run the python script with the specified command and pass any additional arguments
python birdset/train.py --config-path '../projects/uncertainbird/configs' --config-dir 'configs' 'logger=wandb_UncertainBird' "$@"