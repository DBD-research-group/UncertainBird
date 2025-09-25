import logging
from typing import Optional, Tuple

import datasets
import pandas as pd
import tensorflow as tf
import torch
from torch import nn

from birdset.configs import PretrainInfoConfig


from birdset.utils.perch_v2_utils.perch_hoplite.zoo.model_configs import (
    load_model_by_name,
)
import tensorflow as tf
import numpy as np
import pandas as pd


SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class PerchV2Model(nn.Module):
    """
    A PyTorch model for bird vocalization classification, integrating a TensorFlow Hub model.

    Attributes:
        PERCH_TF_HUB_URL (str): URL to the TensorFlow Hub model for bird vocalization.
        EMBEDDING_SIZE (int): The size of the embeddings produced by the TensorFlow Hub model.
        num_classes (int): The number of classes to classify into.
        tfhub_version (str): The version of the TensorFlow Hub model to use.
        train_classifier (bool): Whether to train a classifier on top of the embeddings.
        restrict_logits (bool): Whether to restrict output logits to target classes only.
        dataset_info_path (Optional[str]): Path to the dataset information file for target class filtering.
        model: The loaded TensorFlow Hub model (loaded dynamically).
        classifier (Optional[nn.Linear]): A linear classifier layer on top of the embeddings.
    """

    # Constants for the model URL and embedding size
    PERCH_TF_HUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier"
    EMBEDDING_SIZE = 1280

    def __init__(
        self,
        num_classes: int,
        train_classifier: bool = False,
        restrict_logits: bool = True,
        label_path: Optional[str] = None,
        pretrain_info: Optional[PretrainInfoConfig] = None,
    ) -> None:
        """
        Initializes the PerchModel with configuration for loading the TensorFlow Hub model,
        an optional classifier, and setup for target class restriction based on dataset info.

        Args:
            num_classes: The number of output classes for the classifier.
            tfhub_version: The version identifier of the TensorFlow Hub model to load.
            label_path: Path to a CSV file containing the class information for the Perch model.
            train_classifier: If True, a classifier is added on top of the model embeddings.
            restrict_logits: If True, output logits are restricted to target classes based on dataset info.
        """
        super().__init__()
        self.model = None  # Placeholder for the loaded model
        self.class_mask = None
        self.class_indices = None

        self.num_classes = num_classes
        self.train_classifier = train_classifier
        self.restrict_logits = restrict_logits

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = pretrain_info.hf_name
        else:
            self.hf_path = None
            self.hf_name = None

        self.classifier = nn.Sequential(
            nn.Linear(self.EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

        self.load_model()

    def load_model(self) -> None:
        """
        Load the model from TensorFlow Hub.
        """
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model_choice = "perch_v2_cpu"
        # cudnn_version = (torch.backends.cudnn.version() % 1000) // 100
        # if cudnn_version >= 3:
        #     physical_devices = tf.config.list_physical_devices("GPU")
        #     if len(physical_devices) > 0:
        #         # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #         import os
        #         os.environ["CUDA_VISIBLE_DEVICES"]="0"
        #         tf.config.experimental.set_visible_devices(
        #         physical_devices[0], "GPU"
        #     )
        #         tf.config.optimizer.set_jit(True)
        #         model_choice = 'perch_v2'
        # else:
        #     import os
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        #     model_choice = 'perch_v2_cpu'

        perch_v2 = load_model_by_name(model_choice)

        self.model = perch_v2.embed

        self.class_label_key = "label"

        if self.restrict_logits:
            # Extract the 'ebird2021' column as a list
            pretrain_classlabels = perch_v2.class_list["labels"].classes

            # Load dataset information
            dataset_info = datasets.load_dataset_builder(
                self.hf_path, self.hf_name
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
                label
                for label in dataset_classlabels
                if label not in pretrain_classlabels
            ]
            if missing_labels:
                logging.warning(f"Missing labels in pretrained model: {missing_labels}")

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_values (torch.Tensor): The input tensor for the classifier.
            labels (Optional[torch.Tensor]): The true labels for the input values. Default is None.

        Returns:
            torch.Tensor: The output of the classifier.
        """
        # If there's an extra channel dimension, remove it
        if input_values.dim() > 2:
            input_values = input_values.squeeze(1)

        device = input_values.device  # Get the device of the input tensor

        # Move the tensor to the CPU and convert it to a NumPy array.
        input_values = input_values.cpu().numpy()

        input_values = tf.convert_to_tensor(input_values, dtype=tf.float32)

        results = self.model(input_values)

        if self.train_classifier:
            embeddings = results.embeddings
            embeddings = embeddings.to(device)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(embeddings)
        else:
            output = torch.Tensor(results.logits[self.class_label_key]).squeeze()

        return output


# p = PerchV2Model(14795)
# samples = torch.zeros(160_000)
# logits = p(samples)
