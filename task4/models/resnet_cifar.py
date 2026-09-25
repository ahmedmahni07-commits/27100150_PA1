"""
CIFAR-appropriate ResNet-18 for Task 4. Randomly initialized (never
ImageNet-pretrained -- Vanilla and GCSC both train "from random
initialization" per spec). torchvision's resnet18 architecture with the
ImageNet-style 7x7 stride-2 stem replaced by a 3x3 stride-1 conv, and the
initial max-pool removed, so it operates directly on 32x32 inputs without
collapsing spatial resolution before layer1 (the standard "CIFAR ResNet"
adaptation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetCifarBackbone(nn.Module):
    """
    Exposes forward_pre_layer3 / forward_post_layer3 as two explicit
    stages -- split at the exact point PROSER's manifold-mixup spec
    requires ("after layer2 and before layer3") -- so PROSER can mix
    features at that boundary without duplicating this module's internals.
    forward() is the ordinary full pass (stem -> layer4 -> pooled 512-d
    feature), used by Vanilla/GCSC and by everything that doesn't need
    the intermediate split.
    """

    feature_dim = 512

    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool  # nn.Identity(); kept as an attribute for structural parity
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

    def forward_pre_layer3(self, x: torch.Tensor) -> torch.Tensor:
        """Stem through layer2 -- this is phi_pre(x) = h in the manifold-mixup spec."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_post_layer3(self, h: torch.Tensor) -> torch.Tensor:
        """layer3 through the pooled 512-d penultimate feature f(x)."""
        x = self.layer3(h)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_pre_layer3(x)
        return self.forward_post_layer3(h)
