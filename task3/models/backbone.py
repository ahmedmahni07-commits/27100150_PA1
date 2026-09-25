"""
ResNet-18 feature extractor. Duplicated from task2/models/backbone.py per
Task 3's suggested repository structure (its own models/ dir) -- identical
architecture, so checkpoints trained/saved by either module's state_dict
are interchangeable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(net.children())[:-1])  # drop fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return torch.flatten(feats, 1)

    def freeze_batchnorm_running_stats(self) -> None:
        """
        Same shared BN policy as Task 2: after any .train() call, put ONLY
        BatchNorm modules into eval() mode so running stats stop updating,
        while their affine (gamma/beta) parameters stay trainable.
        """
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
