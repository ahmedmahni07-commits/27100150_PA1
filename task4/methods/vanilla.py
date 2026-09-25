"""
Step 1: Vanilla closed-set baseline -- ten-class CIFAR ResNet-18 trained
from random initialization with plain cross-entropy, standard crop+flip
augmentation only (no RandAugment -- that's GCSC's one and only change,
applied in task4/train.py's choice of transform, not here).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from task4.models.classifier_head import LinearClassifierHead
from task4.models.resnet_cifar import ResNetCifarBackbone


@dataclass
class VanillaModel:
    backbone: ResNetCifarBackbone
    head: LinearClassifierHead

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def train_mode(self) -> None:
        self.backbone.train()
        self.head.train()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def to(self, device: torch.device) -> "VanillaModel":
        self.backbone.to(device)
        self.head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> VanillaModel:
    backbone = ResNetCifarBackbone()
    head = LinearClassifierHead(feature_dim=backbone.feature_dim, num_classes=cfg["data"]["num_classes"])
    model = VanillaModel(backbone=backbone, head=head)
    model.to(device)
    return model


def training_step(
    model: VanillaModel,
    batch,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """
    `batch` is a plain (images, labels) pair from a standard shuffled
    DataLoader -- no domain-balanced construction needed here (unlike
    Tasks 2/3), CIFAR-10 has no domain structure.

    Calls loss.backward() but not optimizer.zero_grad()/step() -- caller
    (task4/train.py) owns the optimizer.
    """
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    features = model.backbone(images)
    logits = model.head(features)
    loss = criterion(logits, labels)
    loss.backward()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()

    return {"loss": loss.item(), "batch_acc": batch_acc}


def save_checkpoint(model: VanillaModel, path: str) -> None:
    torch.save(
        {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
        path,
    )


def load_checkpoint(model: VanillaModel, path: str, device: torch.device) -> VanillaModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model
