from birdset.modules.models.birdset_models.convnext_bs import ConvNextBirdSet
import datasets
import torch
from uncertainbird.modules.models.UncertainBirdModel import UncertrainBirdModel


class ConvNeXtBS(ConvNextBirdSet, UncertrainBirdModel):

    def get_label_mappings(self):

        xcl_labels = (
            datasets.load_dataset_builder("DBD-research-group/BirdSet", "XCL")
            .info.features["ebird_code"]
            .names
        )
        label2id = {label: idx for idx, label in self.config.id2label.items()}

        return {
            "pretrain": label2id,
            "xcl": label2id,
        }

    def forward(self, waveforms: torch.Tensor):
        spectrograms = self.preprocess(waveforms).squeeze(1)
        return self.model(spectrograms)
