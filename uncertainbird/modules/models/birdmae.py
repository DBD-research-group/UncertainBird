import logging
from typing import Optional
import datasets
from torch import nn
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from birdset.configs import PretrainInfoConfig

from uncertainbird.modules.models.UncertainBirdModel import UncertrainBirdModel


class BirdMAE(UncertrainBirdModel):
    """
    AudioProtoPNet model trained on BirdSet XCL dataset.
    The model expects a raw 1-channel 5s waveform with sample rate of 32kHz as input.
    Its preprocess function uses the model's provided Hugging Face feature extractor to:
        - convert the waveform to a mel spectrogram in the format expected by the model
        - (normalization and scaling are handled internally by the feature extractor)
    """

    def __init__(
        self,
        num_classes: int = 9736,  # kept for API symmetry; model's classifier head is fixed by the checkpoint
    ):
        super().__init__()
        # Load model + feature extractor from the checkpoint (requires trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "DBD-research-group/BirdMAE-XCL", trust_remote_code=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/BirdMAE-XCL",
            trust_remote_code=True,
        )
        self.config = self.model.config

    def preprocess(self, waveform: torch.Tensor):
        """
        Convert a raw waveform into the mel spectrogram tensor the model expects.
        Returns:
            - Tensor ready to be fed to self.model.
        """
        input_features = self.feature_extractor(waveform)

        return input_features

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Preprocess raw waveform(s)
        input_values = input_values.squeeze(1)  # remove channel dim if present
        preprocessed = self.preprocess(input_values)
        # preprocessed = preprocessed.squeeze(1)  # remove channel dim if present

        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device

        # Feature extractor may return a dict (HF style) or a tensor
        if isinstance(preprocessed, dict):
            for k, v in preprocessed.items():
                if torch.is_tensor(v):
                    preprocessed[k] = v.to(
                        device=model_device, dtype=model_dtype, non_blocking=True
                    )
        elif torch.is_tensor(preprocessed):
            preprocessed = preprocessed.to(
                device=model_device, dtype=model_dtype, non_blocking=True
            )
        else:
            raise TypeError(
                f"Unsupported preprocessed input type: {type(preprocessed)}"
            )

        # Forward through HF model (supports dict unpacking)
        outputs = (
            self.model(**preprocessed)
            if isinstance(preprocessed, dict)
            else self.model(preprocessed)
        )

        # Return logits tensor (standard HF naming); fallback to raw outputs if absent
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs

    def get_label_mappings(self):
        xcl_labels = (
            datasets.load_dataset_builder("DBD-research-group/BirdSet", "XCL")
            .info.features["ebird_code"]
            .names
        )
        label2id = {label: idx for idx, label in self.config.id2label.items()}
        xcl_map = {lbl: i for i, lbl in enumerate(xcl_labels)}
        return {
            "pretrain": label2id,
            "xcl": xcl_map,
            "pretrain_list": list(self.config.id2label.values()),
            "xcl_list": xcl_labels,
        }
