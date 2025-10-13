#!/usr/bin/env python
"""
Dump model predictions, targets and metrics for one or more BirdSet subsets.

This script mirrors the behavior of the DumpPredictionsCallback by collecting all
predictions (logits mapped to the full 9,736 BirdSet XCL label space), expanding
target multi-hot vectors into the same space, computing metrics, and saving a
pickle artifact per dataset.

Usage example:
    python dump_predictions.py \
        --model perch_v2 \
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
import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


# Local imports (repository specific)
try:
    from uncertainbird.datamodule.BirdSetEvalDataModule import BirdSetEvalDataModule
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
    from uncertainbird.utils.misc import expand_logits, expand_targets
except Exception:
    # Fallback: dummy metrics function
    def print_metrics(preds: torch.Tensor, targets: torch.Tensor):  # type: ignore
        return {"num_samples": preds.shape[0]}


import datasets  # HF datasets

FULL_LABEL_SPACE_SIZE = 9736  # BirdSet XCL total classes
HF_PATH = "DBD-research-group/BirdSet"
XCL_HF_NAME = "XCL"



def process_dataset(dataset_name: str, model, maps, args):
    out_dir = Path(args.output_dir) / args.model / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule setup
    dm = BirdSetEvalDataModule(
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


    # Determine dataset subset labels (order used by targets)
    ds_builder = datasets.load_dataset_builder(HF_PATH, dataset_name)
    ds_labels = ds_builder.info.features["ebird_code"].names

    limit = args.max_samples if args.max_samples is not None else len(test_ds)
    with torch.no_grad():
        for idx in tqdm(range(min(len(test_ds), limit)), desc=f"{dataset_name} samples"):
            sample = test_ds[idx]
            logits = model(sample['input_values'])
            raw_logits_list.append(logits)
            raw_targets_list.append(sample["labels"].unsqueeze(0).cpu())  # (1, D)

    raw_logits = torch.cat(raw_logits_list, dim=0).cpu()
    raw_targets = torch.cat(raw_targets_list, dim=0).cpu()

    # Expand to full space
    full_logits, missing_labels = expand_logits(raw_logits, ds_labels, maps, FULL_LABEL_SPACE_SIZE)
    full_targets = expand_targets(raw_targets, ds_labels, maps['xcl'], FULL_LABEL_SPACE_SIZE)

    if missing_labels:
        print(
            f"Dataset {dataset_name}: {len(missing_labels)} labels missing pretrain space."
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
        description="Dump model predictions over BirdSet subsets."
    )
    p.add_argument(
        "--model",
        type=str,
        choices=["perch_v2", "birdmae", "audioprotopnet", "convnext_bs"],
        required=True,
        help="Model to use",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=False,
        default=["PER", "POW", "NES", "UHH", "HSN", "NBP", "SSW", "SNE"],
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

    # switch model loading based on arg
    print(f"Loading model {args.model} (GPU {args.gpu}) ...")
    if args.model == "perch_v2":
        from uncertainbird.modules.models.perchv2 import Perchv2Model
        model = Perchv2Model(num_classes=FULL_LABEL_SPACE_SIZE, gpu_to_use=args.gpu, map_logits_to_XCL=True)
    if args.model == "audioprotopnet":
        from uncertainbird.modules.models.audioprotopnet import AudioProtoPNet
        model = AudioProtoPNet(pretrain_info=None) 
        # model.to("cuda" if args.gpu is not None else "cpu") 
    if args.model == "birdmae":
        from uncertainbird.modules.models.birdmae import BirdMAE
        model = BirdMAE(num_classes=FULL_LABEL_SPACE_SIZE)
    if args.model == "convnext_bs":
        from uncertainbird.modules.models.convnext_bs import ConvNeXtBS
        model = ConvNeXtBS(num_classes=FULL_LABEL_SPACE_SIZE)

    print("Building label mappings ...")
    maps = model.get_label_mappings()

    for ds in args.datasets:
        print(f"\nProcessing dataset: {ds}")
        process_dataset(ds, model, maps, args)


    print("All done.")


if __name__ == "main__":  # intentional typo safeguard
    main()

if __name__ == "__main__":
    main()
