#!/bin/bash

# A simple script that loops over all specified models and BirdSet datasets and dumps the predictions to:
# /workspace/logs/predictions/<config_path>/<dataset_name>
# 
# Usage: ./projects/uncertainbird/scripts/dump-predictions.sh --config <config_path> [--models <model1,model2,...>] [--datasets <dataset1,dataset2,...>] [--gpu <gpu_id>] [--batch_size <batch_size>] [--output_dir <output_dir>] [--extras <extra_args>]


# Default values
output_dir=/workspace/logs/predictions
config_path="eval_calibration"
default_models=("convnext_bs")
default_dnames=("PER" "POW" "NES" "UHH" "HSN" "NBP" "SSW" "SNE")
default_batch_size=512
gpu=0
extras=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --models)
      IFS=',' read -r -a models <<< "$2"
      shift 2;;
    --datasets)
      IFS=',' read -r -a selected_dnames <<< "$2"
      shift 2;;
    --gpu)
      gpu=$2
      shift 2;;
    --config)
      config_path="$2"
      shift 2;;
    --batch_size)
      batch_size=$2
      shift 2;;
    --output_dir)
      output_dir=$2
      shift 2;;
    --extras)
      extras="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

# Set defaults if not provided
models=("${models[@]:-${default_models[@]}}")
dnames=("${default_dnames[@]}")
batch_size=${batch_size:-$default_batch_size}
output_dir=${output_dir:-$output_dir}


# Validate required argument
if [ -z "$config_path" ]; then
  echo "Error: --config is required"
  echo "Usage: $0 --config <config_path> [--models <model1,model2,...>] [--datasets <dataset1,dataset2,...>] [--gpu <gpu_id>] [--batch_size <batch_size>]"
  exit 1
fi

# Filter datasets if specific ones are given
if [ "${#selected_dnames[@]}" -gt 0 ]; then
  filtered_dnames=()
  filtered_timeouts=()
  for dataset in "${selected_dnames[@]}"; do
    for i in "${!dnames[@]}"; do
      if [ "${dnames[$i]}" == "$dataset" ]; then
        filtered_dnames+=("${dnames[$i]}")
        filtered_timeouts+=("${default_timeouts[$i]}")
      fi
    done
  done
  dnames=("${filtered_dnames[@]}")
  default_timeouts=("${filtered_timeouts[@]}")

fi

# Function to handle Ctrl+C (SIGINT) and decide behavior
trap ' 
  if [ "$first_ctrl_c_triggered" = true ]; then
    echo "Second Ctrl+C detected. Exiting..."
    exit 1
  else
    echo "Ctrl+C detected. Skipping current experiment... Press Ctrl+C again to exit"
    first_ctrl_c_triggered=true
  fi
' SIGINT

# Main loop
for model in "${models[@]}"; do
  echo "Running experiments for model $model"
  for i in "${!dnames[@]}"; do
    dname=${dnames[$i]}
    timeout=${default_timeouts[$i]}
    echo "Running with dataset_name=$dname"

    # Reset quit flag
    sleep 3 # This allows detecting a quick second Ctrl+C press
    first_ctrl_c_triggered=false

    # Build the extra arguments if --extras was provided
    if [ -n "$extras" ]; then
      extra_args=$(echo "$extras" | sed 's/,/ /g' | sed 's/=/=/g')
    fi

    /workspace/projects/uncertainbird/train.sh \
      $timeout \
      experiment="$config_path/$model" \
      datamodule.dataset.dataset_name=$dname \
      datamodule.dataset.hf_name=$dname \
      datamodule.loaders.test.batch_size=$batch_size \
      callbacks.dump_predictions.save_dir="$output_dir/$model/$dname" \
      $extra_args
  done
done