#!/usr/bin/env python
"""
Dump AudioProtoPNet predictions, targets and metrics for one or more BirdSet subsets.

This script mirrors the behavior of the DumpPredictionsCallback by collecting all
predictions (over the full 9,736 BirdSet XCL label space), expanding target multi-hot
vectors into the same space, computing metrics, and saving a pickle artifact per dataset.

Usage example:
    python dump_audioprotopnet_predictions.py \
        --datasets HSN NBP SSW \
        --gpu 0 \
        --output-dir /workspace/logs/predictions/audioprotopnet \

Outputs per dataset (inside <output-dir>/<dataset>/):
    audioprotopnet_test_predictions_<dataset>_<timestamp>.pkl  (predictions, targets, metadata, metrics)

Notes:
    * Uses Hugging Face model "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL".
    * Iterates sample-by-sample (can be batched later if desired).
"""
from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Local imports (repository specific)
try:
    from birdset.datamodule.birdset_datamodule import BirdSetDataModule
    from birdset.datamodule.base_datamodule import (
        DatasetConfig,
        BirdSetTransformsWrapper,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import BirdSetDataModule components. Ensure project dependencies are installed."
    ) from e

try:
    from uncertainbird.utils.plotting import print_metrics
except Exception:
    # Fallback: dummy metrics function
    def print_metrics(preds: torch.Tensor, targets: torch.Tensor):  # type: ignore
        return {"num_samples": preds.shape[0]}


import datasets  # HF datasets
import pandas as pd

FULL_LABEL_SPACE_SIZE = 9736  # BirdSet XCL total classes
XCL_HF_NAME = "XCL"
HF_PATH = "DBD-research-group/BirdSet"
MODEL_NAME = "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL"


def load_model(gpu: int | None):
    """Load AudioProtoPNet HF model and move to device."""
    if gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from uncertainbird.modules.models.audioprotopnet import AudioProtoPNet

    model = AudioProtoPNet()
    model.eval().to(device)
    return model, device


def load_xcl_labels() -> List[str]:
    """Return ordered XCL label list from HF dataset builder."""
    xcl_labels = (
        datasets.load_dataset_builder(HF_PATH, XCL_HF_NAME)
        .info.features["ebird_code"]
        .names
    )
    if len(xcl_labels) != FULL_LABEL_SPACE_SIZE:
        print(
            f"Warning: XCL label space size {len(xcl_labels)} != expected {FULL_LABEL_SPACE_SIZE}"
        )
    return xcl_labels


def identity_logits(logits: torch.Tensor) -> torch.Tensor:
    """AudioProtoPNet already outputs logits over full XCL space (9736)."""
    return logits


def expand_targets(
    targets: torch.Tensor, dataset_labels: List[str], maps: Dict[str, Dict[str, int]]
):
    """Expand dataset multi-hot targets (N, D) into full XCL space (N, 9736)."""
    xcl_map = maps["xcl"]
    N = targets.shape[0]
    full = torch.zeros(N, FULL_LABEL_SPACE_SIZE, dtype=targets.dtype)
    for j, lbl in enumerate(dataset_labels):
        xi = xcl_map.get(lbl)
        if xi is None:
            continue
        full[:, xi] = targets[:, j]
    return full


def process_dataset(
    dataset_name: str,
    model: torch.nn.Module,
    device: torch.device,
    xcl_labels: List[str],
    args,
):
    out_dir = Path(args.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule setup
    dm = BirdSetDataModule(
        dataset=DatasetConfig(
            data_dir=args.data_dir,
            hf_path=HF_PATH,
            hf_name=dataset_name,
            n_workers=args.num_workers,
            val_split=0.0001,
            task="multilabel",
            classlimit=None,  # keep full set for subset
            eventlimit=None,
            sample_rate=32_000,
        ),
        transforms=BirdSetTransformsWrapper(
            sample_rate=32_000,
            model_type="waveform",
        ),
    )
    dm.prepare_data()
    dm.setup("test")
    test_ds = dm.test_dataset

    if test_ds is None or len(test_ds) == 0:
        print(f"Warning: No test samples for dataset {dataset_name}. Skipping.")
        return

    # Collect logits (full space) & raw targets (subset space)
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    # Determine dataset subset labels (order used by targets)
    ds_builder = datasets.load_dataset_builder(HF_PATH, dataset_name)
    ds_labels = ds_builder.info.features["ebird_code"].names

    limit = args.max_samples if args.max_samples is not None else len(test_ds)

    for idx in tqdm(range(min(len(test_ds), limit)), desc=f"{dataset_name} samples"):
        sample = test_ds[idx]
        wav = sample["input_values"]  # (1, T)
        with torch.no_grad():
            wav = wav.to(device)
            logits = model(wav)  # (1, 9736)
        logits_list.append(logits.cpu())
        targets_list.append(sample["labels"].unsqueeze(0).cpu())  # (1, D)

    full_logits = identity_logits(torch.cat(logits_list, dim=0))
    raw_targets = torch.cat(targets_list, dim=0)
    full_targets = expand_targets(
        raw_targets, ds_labels, {"xcl": {lbl: i for i, lbl in enumerate(xcl_labels)}}
    )
    missing_labels = []  # Not applicable (model covers full space)

    # Convert to probabilities (softmax over non-zero columns? Use softmax over all for consistency)
    predictions = torch.softmax(full_logits, dim=1)

    # Metrics
    metrics = print_metrics(predictions, full_targets.to(torch.int))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle_name = f"audioprotopnet_test_predictions_{dataset_name}_{timestamp}.pkl"

    metadata = {
        "total_samples": predictions.shape[0],
        "prediction_shape": list(predictions.shape),
        "target_shape": list(full_targets.shape),
        "num_batches": 1,
        "dataset": dataset_name,
        "missing_labels_count": 0,
        "missing_labels_sample": [],
        "model_info": {
            "model_source": MODEL_NAME,
            "class_name": "AudioProtoPNet",
            "num_classes": FULL_LABEL_SPACE_SIZE,
        },
        "metrics_keys": list(metrics.keys()),
    }

    data = {
        "logits": full_logits,
        "predictions": predictions,
        "targets": full_targets,
        "metadata": metadata,
        "metrics": metrics,
    }

    # Save artifacts
    with open(out_dir / pickle_name, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved dataset {dataset_name}: {pickle_name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Dump AudioProtoPNet predictions over BirdSet subsets."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of BirdSet subset keys (e.g. HSN NBP SSW)",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory to write per-dataset results",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/data_birdset",
        help="Local cache directory for BirdSet data",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loading workers for BirdSetDataModule",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for number of test samples (debug)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Defer setting GPU until now
    if args.gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    print(f"Loading AudioProtoPNet model (GPU {args.gpu}) ...")
    model, device = load_model(args.gpu)

    print("Loading XCL label list ...")
    xcl_labels = load_xcl_labels()

    for ds in args.datasets:
        print(f"\nProcessing dataset: {ds}")
        try:
            process_dataset(ds, model, device, xcl_labels, args)
        except Exception as e:
            print(f"Error processing {ds}: {e}")

    print("All done.")


if __name__ == "main__":  # intentional typo safeguard
    main()

if __name__ == "__main__":
    main()
