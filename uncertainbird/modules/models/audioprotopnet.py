from typing import Optional
from torch import nn
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from birdset.configs import PretrainInfoConfig

from uncertainbird.modules.models.UncertainBirdModel import UncertrainBirdModel


class AudioProtoPNet(UncertrainBirdModel):
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
            "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL",
            trust_remote_code=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL",
            trust_remote_code=True,
        )
        self.config = self.model.config
        self.pretrain_info = pretrain_info

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
        preprocessed = self.preprocess(input_values)
        preprocessed = preprocessed.squeeze(2)  # remove channel dim if present

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
        return {
            "pretrain": {lbl: i for i, lbl in enumerate(self.config.id2label.values())},
            "xcl": {lbl: i for i, lbl in enumerate(self.config.id2label.values())},
        }
