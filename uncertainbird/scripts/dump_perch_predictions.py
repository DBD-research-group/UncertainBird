#!/usr/bin/env python
"""
Dump Perch v2 predictions, targets and metrics for one or more BirdSet subsets.

This script mirrors the behavior of the DumpPredictionsCallback by collecting all
predictions (logits mapped to the full 9,736 BirdSet XCL label space), expanding
target multi-hot vectors into the same space, computing metrics, and saving a
pickle artifact per dataset.

Usage example:
    python dump_perch_predictions.py \
        --datasets HSN NBP SSW \
        --gpu 0 \
        --output-dir /workspace/logs/predictions/perch_v2 \

Outputs per dataset (inside <output-dir>/<dataset>/):
    test_predictions_<timestamp>.pkl  (contains predictions, targets, metadata, metrics)
    logits.pt                         (torch.Tensor of shape (N, 9736))
    targets.pt                        (torch.Tensor of shape (N, 9736))
    metrics.json                      (JSON of computed metrics)

Notes:
    * Requires TensorFlow + tensorflow_hub for the Perch v2 model.
    * Assumes resources/perch_v2_ebird_classes.csv exists (as in repo) with column 'ebird2021'.
    * Iterates sample-by-sample (can be optimized with batching if needed).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Ensure TF GPU selection BEFORE importing tensorflow (set CUDA device mask)
# (We set this later after parsing args if needed.)
import tensorflow_hub as hub  # type: ignore
import tensorflow as tf  # type: ignore

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
PERCH_TF_HUB_HANDLE = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
PERCH_CLASS_CSV = Path(
    "/workspace/projects/uncertainbird/resources/perch_v2_ebird_classes.csv"
)


def load_perch_model(gpu: int | None):
    if gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    tf.keras.backend.clear_session()
    # set tf gpu to gpu
    physical_devices = tf.config.list_physical_devices("GPU")
    if gpu is not None and len(physical_devices) > gpu:
        tf.config.experimental.set_visible_devices(physical_devices[gpu], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[gpu], True)
    elif physical_devices:
        tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)
    model = hub.load(PERCH_TF_HUB_HANDLE)
    return model


def build_label_mappings() -> Dict[str, Dict[str, int]]:
    """Build mapping dicts for fast label index resolution.

    Returns:
        dict with keys: 'pretrain', 'xcl'
    """
    if not PERCH_CLASS_CSV.exists():
        raise FileNotFoundError(f"Perch class CSV not found: {PERCH_CLASS_CSV}")
    pretrain_df = pd.read_csv(PERCH_CLASS_CSV)
    if "ebird2021" not in pretrain_df.columns:
        raise KeyError("Expected column 'ebird2021' in perch class CSV.")
    pretrain_labels = pretrain_df["ebird2021"].tolist()
    pretrain_map = {lbl: i for i, lbl in enumerate(pretrain_labels)}

    xcl_labels = (
        datasets.load_dataset_builder(HF_PATH, XCL_HF_NAME)
        .info.features["ebird_code"]
        .names
    )
    if len(xcl_labels) != FULL_LABEL_SPACE_SIZE:
        print(
            f"Warning: XCL label space size {len(xcl_labels)} != expected {FULL_LABEL_SPACE_SIZE}"
        )
    xcl_map = {lbl: i for i, lbl in enumerate(xcl_labels)}

    return {
        "pretrain": pretrain_map,
        "xcl": xcl_map,
        "pretrain_list": pretrain_labels,  # store lists for ordered indexing
        "xcl_list": xcl_labels,
    }


def expand_logits(
    logits: torch.Tensor, dataset_labels: List[str], maps: Dict[str, Dict[str, int]]
):
    """Expand subset logits to full XCL space.

    Args:
        logits: (N, P) Perch logits over its pretrain class subset order.
        dataset_labels: labels (ordered) for the dataset subset.
        maps: mapping dict from build_label_mappings.

    Returns:
        full_logits: (N, 9736)
        missing_labels: labels in dataset not present in perch pretrain list
    """
    pretrain_map = maps["pretrain"]
    xcl_map = maps["xcl"]

    N = logits.shape[0]
    full = torch.zeros(N, FULL_LABEL_SPACE_SIZE, dtype=logits.dtype)

    missing = []
    # Build list once for index gather
    gather_indices = []
    target_positions = []
    for lbl in dataset_labels:
        pi = pretrain_map.get(lbl)
        if pi is None:
            missing.append(lbl)
            continue
        xi = xcl_map.get(lbl)
        if xi is None:
            missing.append(lbl)
            continue
        gather_indices.append(pi)
        target_positions.append(xi)

    if gather_indices:
        gather_t = torch.as_tensor(gather_indices, dtype=torch.long)
        subset_logits = logits[:, gather_t]  # (N, K)
        dest_idx = torch.as_tensor(target_positions, dtype=torch.long)
        full[:, dest_idx] = subset_logits

    return full, missing


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


def process_dataset(dataset_name: str, model, maps, args):
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

    # Collect raw logits
    raw_logits_list = []  # over perch pretrain space
    raw_targets_list = []  # dataset subset space

    serving_fn = model.signatures["serving_default"]

    # Determine dataset subset labels (order used by targets)
    ds_builder = datasets.load_dataset_builder(HF_PATH, dataset_name)
    ds_labels = ds_builder.info.features["ebird_code"].names

    limit = args.max_samples if args.max_samples is not None else len(test_ds)

    for idx in tqdm(range(min(len(test_ds), limit)), desc=f"{dataset_name} samples"):
        sample = test_ds[idx]
        wav = sample["input_values"].squeeze(0).detach().cpu().numpy()  # (T,)
        tf_in = tf.convert_to_tensor(wav[np.newaxis, :], dtype=tf.float32)
        out = serving_fn(inputs=tf_in)
        logits_np = out["label"].numpy()  # (1, P)
        raw_logits_list.append(torch.from_numpy(logits_np))
        raw_targets_list.append(sample["labels"].unsqueeze(0).cpu())  # (1, D)

    raw_logits = torch.cat(raw_logits_list, dim=0)
    raw_targets = torch.cat(raw_targets_list, dim=0)

    # Expand to full space
    full_logits, missing_labels = expand_logits(raw_logits, ds_labels, maps)
    full_targets = expand_targets(raw_targets, ds_labels, maps)

    if missing_labels:
        print(
            f"Dataset {dataset_name}: {len(missing_labels)} labels missing from Perch pretrain space."
        )

    # Convert to probabilities (softmax over non-zero columns? Use softmax over all for consistency)
    predictions = torch.softmax(full_logits, dim=1)

    # Metrics
    metrics = print_metrics(predictions, full_targets.to(torch.int))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle_name = f"perch_test_predictions_{dataset_name}_{timestamp}.pkl"

    metadata = {
        "total_samples": predictions.shape[0],
        "prediction_shape": list(predictions.shape),
        "target_shape": list(full_targets.shape),
        "num_batches": 1,
        "dataset": dataset_name,
        "missing_labels_count": len(missing_labels),
        "missing_labels_sample": missing_labels[:25],
        "model_info": {
            "model_source": PERCH_TF_HUB_HANDLE,
            "class_name": "PerchV2TFHub",
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
        description="Dump Perch v2 predictions over BirdSet subsets."
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

    print(f"Loading Perch model on GPU {args.gpu} ...")
    model = load_perch_model(args.gpu)

    print("Building label mappings ...")
    maps = build_label_mappings()

    for ds in args.datasets:
        print(f"\nProcessing dataset: {ds}")
        try:
            process_dataset(ds, model, maps, args)
        except Exception as e:
            print(f"Error processing {ds}: {e}")

    print("All done.")


if __name__ == "main__":  # intentional typo safeguard
    main()

if __name__ == "__main__":
    main()
