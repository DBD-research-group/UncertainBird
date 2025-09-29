import logging
from typing import Optional
import datasets
from torch import nn
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from birdset.configs import PretrainInfoConfig


class BirdMAE(nn.Module):
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
        pretrain_info: Optional[PretrainInfoConfig] = None,
    ):
        super().__init__()
        # Load model + feature extractor from the checkpoint (requires trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "DBD-research-group/BirdMAE-XCL",
            trust_remote_code=True,
            num_labels=9736,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/BirdMAE-XCL",
            trust_remote_code=True,
        )
        self.config = self.model.config
        self.pretrain_info = pretrain_info

        # load class_mask as labels might not match exactly
        # Load the class list from the CSV file
        pretrain_classlabels = list(self.config.id2label.values())

        # Load dataset information
        dataset_info = datasets.load_dataset_builder(
            self.pretrain_info.hf_path, self.pretrain_info.hf_name
        ).info
        dataset_classlabels = dataset_info.features["ebird_code"].names

        # Create the class mask
        self.class_mask = [
            pretrain_classlabels.index(label)
            for label in dataset_classlabels
            if label in pretrain_classlabels
        ]
        self.class_indices = [
            i
            for i, label in enumerate(dataset_classlabels)
            if label in pretrain_classlabels
        ]

        # Log missing labels
        missing_labels = [
            label for label in dataset_classlabels if label not in pretrain_classlabels
        ]
        if missing_labels:
            logging.warning(f"Missing labels in pretrained model: {missing_labels}")

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
            logits = outputs.logits
            if self.class_mask:
                # Initialize full_logits to 0 for all classes
                full_logits = torch.full(
                    (
                        logits.shape[0],
                        9736,
                    ),  # Assuming 9736 is the total number of classes
                    0.0,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                # Extract valid logits using indices from class_mask and directly place them
                full_logits[:, self.class_indices] = logits[:, self.class_mask]
                logits = full_logits

            return logits
        return outputs
