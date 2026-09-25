"""
Step 1: Source-only ERM baseline. Trained once here; this exact checkpoint
is reused unchanged as the Task 3 ERM baseline (do not retrain it there).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifierHead


@dataclass
class SourceOnlyModel:
    backbone: ResNet18Backbone
    head: LinearClassifierHead

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def train_mode(self) -> None:
        """
        Put both submodules in train() mode, then immediately reapply the
        BN-freeze policy — .train() flips BatchNorm back to updating its
        running stats, so this has to be reasserted every time.
        """
        self.backbone.train()
        self.head.train()
        self.backbone.freeze_batchnorm_running_stats()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def to(self, device: torch.device) -> "SourceOnlyModel":
        self.backbone.to(device)
        self.head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> SourceOnlyModel:
    backbone = ResNet18Backbone(pretrained=True)
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=cfg["model"]["num_classes"],
    )
    model = SourceOnlyModel(backbone=backbone, head=head)
    model.to(device)
    return model


def training_step(
    model: SourceOnlyModel,
    source_batch: dict,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    progress_p: float = 0.0,  # unused here; present so train.py's call is uniform across methods
) -> dict:
    """
    One forward/backward pass on a domain-balanced source batch (the
    target images in the batch dict, if present, are unused for this
    method). Calls loss.backward() but NOT optimizer.zero_grad() or
    optimizer.step() — train.py owns the optimizer so it can apply the
    same step/clip/schedule logic uniformly across all four methods.

    Caller contract: optimizer.zero_grad() before calling this,
    optimizer.step() after.
    """
    images = source_batch["source_images"].to(device, non_blocking=True)
    labels = source_batch["source_labels"].to(device, non_blocking=True)

    features = model.backbone(images)
    logits = model.head(features)
    loss = criterion(logits, labels)
    loss.backward()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()

    return {"loss": loss.item(), "cls_loss": loss.item(), "batch_acc": batch_acc}


def save_checkpoint(model: SourceOnlyModel, path: str) -> None:
    """
    Checkpoint format: {"backbone": state_dict, "head": state_dict}.
    Keep this exact key naming — Task 3's ERM baseline loader should load
    this file directly rather than reimplementing the format.
    """
    torch.save(
        {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
        path,
    )


def load_checkpoint(model: SourceOnlyModel, path: str, device: torch.device) -> SourceOnlyModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model