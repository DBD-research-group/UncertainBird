import torch


class UncertrainBirdModel(torch.nn.Module):

    def get_label_mappings(self):
        raise NotImplementedError("This is an abstract base class.")
    
    def preprocess(self, waveform: torch.Tensor):
        raise NotImplementedError("This is an abstract base class.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is an abstract base class.")