import torch
import torch.nn as nn


class LinearClassifierHead(nn.Module):
    """Single linear layer: 512-d feature -> 7 class logits."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)