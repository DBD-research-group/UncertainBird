import os
from pathlib import Path
from typing import Dict

import datasets
import numpy as np
import pandas as pd
import torch

# TensorFlow and tensorflow_hub are heavy and may not be installed in
# all developer environments (and the language server will flag them if
# imported at module import time). Import them lazily where needed so
# the rest of the package can be inspected without requiring TF.

from uncertainbird.modules.models.UncertainBirdModel import UncertrainBirdModel

HF_PATH = "DBD-research-group/BirdSet"
PERCH_TF_HUB_HANDLE = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
PERCH_CLASS_CSV = Path("/workspace/uncertainbird/resources/perch_v2_ebird_classes.csv")
XCL_HF_NAME = "XCL"


def load_perch_model(gpu: int | None):
    # lazy imports to avoid hard dependency at module import time
    try:
        import importlib

        tf = importlib.import_module("tensorflow")
        hub = importlib.import_module("tensorflow_hub")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow and tensorflow_hub are required to load the Perch model. "
            "Install tensorflow and tensorflow-hub (or use a different model class)"
        ) from exc

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


def build_label_mappings(
    full_label_space_size: int = 9736,
) -> Dict[str, Dict[str, int]]:
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
    if len(xcl_labels) != full_label_space_size:
        print(
            f"Warning: XCL label space size {len(xcl_labels)} != expected {full_label_space_size}"
        )
    xcl_map = {lbl: i for i, lbl in enumerate(xcl_labels)}

    return {
        "pretrain": pretrain_map,
        "xcl": xcl_map,
        "pretrain_list": pretrain_labels,  # store lists for ordered indexing
        "xcl_list": xcl_labels,
    }


class Perchv2Model(UncertrainBirdModel):

    def __init__(
        self,
        num_classes: int = 9736,  # kept for API symmetry; model's classifier head is fixed by the checkpoint
        gpu_to_use: int | None = None,
    ):
        super().__init__()
        # Load model + feature extractor from the checkpoint (requires trust_remote_code=True)
        self.model = load_perch_model(gpu_to_use)
        self.label_mapping = build_label_mappings(num_classes)
        self.serving_fn = self.model.signatures["serving_default"]

    def get_label_mappings(self):
        return self.label_mapping

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run the model on a batch of waveforms.

        Args:
            waveform: (N, T) float tensor, raw waveform at 32kHz

        Returns:
            logits: (N, 9736) float tensor, raw logits over full XCL space
        """
        # lazy import here as well to keep module import light-weight
        try:
            import importlib

            tf = importlib.import_module("tensorflow")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TensorFlow is required to run Perchv2Model.forward. "
                "Install tensorflow to use this model."
            ) from exc

        wav = waveform.squeeze(1).detach().cpu().numpy()  # (T,)
        tf_in = tf.convert_to_tensor(wav, dtype=tf.float32)
        out = self.serving_fn(inputs=tf_in)
        logits_np = out["label"].numpy()
        logits = torch.from_numpy(logits_np)  # (1, P)

        return logits

    def transform_logits_to_probs(self, logits):
        return torch.softmax(logits, dim=1)