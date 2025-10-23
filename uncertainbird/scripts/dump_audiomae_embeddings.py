#!/usr/bin/env python
"""
Dump AudioMAE embeddings (clip-level and frame-level) and logits for BirdSet subsets.

This script loads the Hugging Face AudioMAE model via timm and iterates over the
BirdSet subsets using the repository's BirdSetEvalDataModule. It saves, per subset
and per split (train and test_5s), the following tensors:

  - clip_embeddings.pt   (shape: [N, 768])
  - frame_embeddings.pt  (shape: [N, 64, 768])   # 1024/16 time patches, mean-pooled across mel
  - logits.pt            (shape: [N, 768])       # identical to clip embeddings (no classifier)

Optionally, a single pickle is also written per split containing the same tensors
and some metadata for convenience.

Example usage:
    python dump_audiomae_embeddings.py \
        --datasets HSN NBP SSW \
        --gpu 0 \
        --output-dir /workspace/logs/embeddings/audiomae

Notes:
  * AudioMAE expects 16 kHz audio and a 1024x128 fbank input. This script will
    resample via the datamodule (set sample_rate=16_000) and then pad/truncate
    each fbank to exactly 1024 frames as per reference snippet provided.
  * No classifier head is attached, so "logits" are stored as the clip-level
    embedding for compatibility with downstream code expecting a logits tensor.
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.compliance import kaldi


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


MEAN = -4.2677393
STD = 4.5689974


@dataclass
class SplitArtifacts:
    clip_embeddings: torch.Tensor  # [N, 768]
    frame_embeddings: torch.Tensor  # [N, 64, 768]
    logits: torch.Tensor  # [N, 768] (same as clip_embeddings)
    total_samples: int


def build_model(device: torch.device) -> torch.nn.Module:
    """Create the AudioMAE model from Hugging Face Hub via timm and set to eval mode."""
    model = timm.create_model(
        "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True
    )
    model = model.to(device).eval()
    return model


def waveform_to_fbank_1024x128(wave: torch.Tensor) -> torch.Tensor:
    """Convert a mono 16 kHz waveform (T,) to (1024, 128) Kaldi fbank with padding/truncation."""
    # torchaudio.compliance.kaldi.fbank expects shape (channel, num_samples)
    wave = wave.detach().cpu().unsqueeze(0)  # (1, T)
    melspec = kaldi.fbank(
        wave,
        htk_compat=True,
        window_type="hanning",
        num_mel_bins=128,
        sample_frequency=16000,
    )  # (n_frames, 128)
    n = melspec.shape[0]
    if n < 1024:
        melspec = F.pad(melspec, (0, 0, 0, 1024 - n))
    else:
        melspec = melspec[:1024]
    # Normalize per AudioMAE recipe
    melspec = (melspec - MEAN) / (STD * 2)
    return melspec  # (1024, 128)


def batch_fbank_from_waveforms(batch_wav: torch.Tensor) -> torch.Tensor:
    """Compute a batch of (B, 1, 1024, 128) fbanks from (B, T) waveforms (CPU ops)."""
    fbanks = [waveform_to_fbank_1024x128(w) for w in batch_wav]
    fb = torch.stack(fbanks, dim=0)  # (B, 1024, 128)
    fb = fb.view(fb.shape[0], 1, 1024, 128)
    return fb


def forward_audiomae(model: torch.nn.Module, fb_batch: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run AudioMAE on a batch of fbanks -> (clip_emb [B,768], frame_emb [B,64,768])."""
    with torch.no_grad():
        fb_batch = fb_batch.to(device)
        # Clip-level embedding from model(x)
        clip_emb = model(fb_batch)  # (B, 768)

        # Frame-level embeddings via forward_features -> patches -> pool across mel dimension
        feats = model.forward_features(fb_batch)  # (B, 513, 768)
        feats = feats[:, 1:]  # remove CLS -> (B, 512, 768)
        feats = feats.unflatten(1, (1024 // 16, 128 // 16))  # (B, 64, 8, 768)
        frame_emb = feats.mean(2)  # (B, 64, 768)

    return clip_emb.detach().cpu(), frame_emb.detach().cpu()


def process_subset(
    subset_name: str,
    model: torch.nn.Module,
    device: torch.device,
    args,
) -> None:
    """Process one BirdSet subset and dump artifacts for train and test splits."""
    base_out = Path(args.output_dir) / subset_name
    base_out.mkdir(parents=True, exist_ok=True)

    # Prepare data module for multilabel with 16kHz waveform output
    dm = BirdSetEvalDataModule(
        dataset=DatasetConfig(
            data_dir=args.data_dir,
            hf_path="DBD-research-group/BirdSet",
            hf_name=subset_name,
            n_workers=args.num_workers,
            val_split=0.0001,
            task="multilabel",
            classlimit=None,
            eventlimit=None,
            sample_rate=16_000,
        ),
        transforms=BirdSetTransformsWrapper(
            sample_rate=16_000,
            model_type="waveform",
        ),
    )
    dm.prepare_data()
    dm.setup("test")  # populates train and test (mapped from test_5s)

    datasets = {"train": dm.train_dataset, "test": dm.test_dataset}

    for split_name, ds in datasets.items():
        if ds is None or len(ds) == 0:
            print(f"Warning: No samples for subset {subset_name} split {split_name}. Skipping.")
            continue

        out_dir = base_out / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        clip_list: List[torch.Tensor] = []
        frame_list: List[torch.Tensor] = []

        for batch in dl:
            wav = batch["input_values"]  # (B, T) 16 kHz
            # Compute fbank on CPU, then send to device once stacked
            fb = batch_fbank_from_waveforms(wav)
            clip_emb, frame_emb = forward_audiomae(model, fb, device)
            clip_list.append(clip_emb)
            frame_list.append(frame_emb)

        clip_embeddings = torch.cat(clip_list, dim=0)
        frame_embeddings = torch.cat(frame_list, dim=0)
        logits = clip_embeddings.clone()  # no classifier; store clip embedding as logits for convenience

        # Save individual tensors
        torch.save(clip_embeddings, out_dir / "clip_embeddings.pt")
        torch.save(frame_embeddings, out_dir / "frame_embeddings.pt")
        torch.save(logits, out_dir / "logits.pt")

        # Metadata + pickle bundle (optional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = {
            "subset": subset_name,
            "split": split_name,
            "total_samples": int(clip_embeddings.shape[0]),
            "clip_embedding_shape": list(clip_embeddings.shape),
            "frame_embedding_shape": list(frame_embeddings.shape),
            "logits_shape": list(logits.shape),
            "model": "gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        }
        bundle = {
            "clip_embeddings": clip_embeddings,
            "frame_embeddings": frame_embeddings,
            "logits": logits,
            "metadata": meta,
        }
        with open(out_dir / f"audiomae_embeddings_{timestamp}.pkl", "wb") as f:
            pickle.dump(bundle, f)

        print(
            f"Saved subset {subset_name} split {split_name}: N={meta['total_samples']} -> {out_dir}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Dump AudioMAE embeddings for BirdSet subsets.")
    p.add_argument(
        "--datasets",
        nargs="+",
        required=False,
        default=["PER", "POW", "NES", "UHH", "HSN", "NBP", "SSW", "SNE"],
        help="List of BirdSet subset keys (e.g., HSN NBP SSW).",
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
        help="Base directory to write per-subset results",
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
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for DataLoader",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Configure device
    if args.gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Build model
    print("Loading AudioMAE model ...")
    model = build_model(device)

    # Process each subset
    for subset in args.datasets:
        print(f"\nProcessing subset: {subset}")
        process_subset(subset, model, device, args)

    print("All done.")


if __name__ == "__main__":
    main()
