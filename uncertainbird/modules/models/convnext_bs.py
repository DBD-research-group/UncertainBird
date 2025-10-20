from birdset.modules.models.birdset_models.convnext_bs import ConvNextBirdSet
import datasets
import torch
from uncertainbird.modules.models.UncertainBirdModel import UncertrainBirdModel
HF_PATH = "DBD-research-group/BirdSet"
XCL_HF_NAME = "XCL"

class ConvNeXtBS(ConvNextBirdSet, UncertrainBirdModel):

    def get_label_mappings(self):
        xcl_labels = (
        datasets.load_dataset_builder(HF_PATH, XCL_HF_NAME)
        .info.features["ebird_code"]
        .names
        )
        xcl_map = {lbl: i for i, lbl in enumerate(xcl_labels)}

        return {
            "pretrain": {lbl: i for i, lbl in self.config.id2label.items()},
            "xcl": xcl_map,
        }
    
    def forward(self, waveforms: torch.Tensor):
        spectrograms = self.preprocess(waveforms).squeeze(1)
        return self.model(spectrograms)
