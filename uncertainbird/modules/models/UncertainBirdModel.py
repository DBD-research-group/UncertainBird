import torch


class UncertrainBirdModel(torch.nn.Module):

    def get_label_mappings(self):
        raise NotImplementedError("This is an abstract base class.")

    def preprocess(self, waveform: torch.Tensor):
        raise NotImplementedError("This is an abstract base class.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is an abstract base class.")
    
    def transform_logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Transform raw logits to probabilities over the full XCL label space.

        Args:
            logits: (N, C) float tensor, raw logits over C classes

        Returns:
            probs: (N, 9736) float tensor, probabilities over full XCL space
        """
        return torch.sigmoid(logits)
