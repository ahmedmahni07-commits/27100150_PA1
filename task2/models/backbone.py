"""
ResNet-18 feature extractor shared by all four Task 2 methods (and reused
unmodified by Task 3).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    torchvision ResNet-18 with ImageNet1K_V1 weights, classifier stripped,
    exposing the 512-d pooled feature.
    """

    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        # Keep everything up to (and including) avgpool; drop the 1000-way fc.
        self.encoder = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 512) pooled features."""
        feats = self.encoder(x)        # (B, 512, 1, 1)
        return torch.flatten(feats, 1) # (B, 512)

    def freeze_batchnorm_running_stats(self) -> None:
        """
        Required policy for every method in Tasks 2 and 3: after any
        `self.train()` call, put ONLY BatchNorm modules into eval() mode so
        running_mean/running_var stop updating, while their affine
        (gamma/beta) parameters stay trainable and everything else stays
        in train() mode.

        Call this once right after every `model.train()` in the training
        loop — `.train()` on the parent module flips BN submodules back to
        train mode too, so this has to be re-applied each time, not just at
        construction.
        """
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()