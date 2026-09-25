"""
Step: ERM baseline for Task 3 = Task 2's source-only checkpoint, reused
UNCHANGED. Per spec, this is not retrained here -- it's simply loaded and
re-evaluated under Task 3's protocol (source-val numbers for the record,
Sketch numbers only in evaluate_sketch.py).

This module deliberately has no training_step -- task3/train.py
special-cases method == "erm" to skip the epoch loop entirely, so
evaluate_sketch.py can still treat erm/dan_dg/sam uniformly afterward (same
checkpoint format, same {backbone, head} keys as every other method).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import LinearClassifierHead


@dataclass
class ERMModel:
    backbone: ResNet18Backbone
    head: LinearClassifierHead

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def train_mode(self) -> None:
        self.backbone.train()
        self.head.train()
        self.backbone.freeze_batchnorm_running_stats()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def to(self, device: torch.device) -> "ERMModel":
        self.backbone.to(device)
        self.head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> ERMModel:
    backbone = ResNet18Backbone(pretrained=True)
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=cfg["model"]["num_classes"],
    )
    model = ERMModel(backbone=backbone, head=head)
    model.to(device)
    return model


def load_source_checkpoint(model: ERMModel, cfg: dict, device: torch.device) -> ERMModel:
    """
    Loads Task 2's frozen source-only checkpoint straight into a Task 3
    ERMModel -- the two architectures are byte-for-byte identical
    (task3/models is a deliberate duplicate of task2/models), so the
    state_dicts load with no remapping.
    """
    path = cfg["erm"]["source_checkpoint"]
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model


def save_checkpoint(model: ERMModel, path: str) -> None:
    torch.save(
        {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
        path,
    )


def load_checkpoint(model: ERMModel, path: str, device: torch.device) -> ERMModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model
