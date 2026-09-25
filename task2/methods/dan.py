"""
Step 2: DAN — MMD alignment on the 512-d feature immediately before the
classifier head.

L_DAN = L_cls(source) + lambda_mmd * MMD^2(features_source, features_target)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from common.mmd import mmd2
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifierHead


@dataclass
class DANModel:
    backbone: ResNet18Backbone
    head: LinearClassifierHead
    lambda_mmd: float
    kernel_bandwidth_multipliers: tuple[float, ...]

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def train_mode(self) -> None:
        self.backbone.train()
        self.head.train()
        self.backbone.freeze_batchnorm_running_stats()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def to(self, device: torch.device) -> "DANModel":
        self.backbone.to(device)
        self.head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> DANModel:
    """
    Constructs backbone -> head in the same order as source_only.build_model,
    from the same post-set_all_seeds RNG state, so the classifier head's
    random init matches source_only's exactly. DAN adds no extra learnable
    modules (no discriminator), which makes it the simplest of the three
    adaptation methods to keep aligned with the shared baseline.
    """
    backbone = ResNet18Backbone(pretrained=True)
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=cfg["model"]["num_classes"],
    )
    model = DANModel(
        backbone=backbone,
        head=head,
        lambda_mmd=cfg["dan"]["lambda_mmd"],
        kernel_bandwidth_multipliers=tuple(cfg["dan"]["kernel_bandwidth_multipliers"]),
    )
    model.to(device)
    return model


def training_step(
    model: DANModel,
    batch: dict,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    progress_p: float = 0.0,  # unused here; present so train.py's call is uniform across methods
) -> dict:
    """
    `batch` is what common.pacs_protocol.domain_balanced_batches yields:
    {"source_images", "source_labels", "source_domain_ids", "target_images", ...}
    Target labels are never read here — domain_balanced_batches doesn't even
    hand them to this function, only the images.

    - forward source_images -> features_s -> logits -> cls_loss (uses source_labels)
    - forward target_images -> features_t (unlabeled)
    - mmd_loss = mmd2(features_s, features_t, model.kernel_bandwidth_multipliers)
    - total = cls_loss + model.lambda_mmd * mmd_loss

    Calls loss.backward() but not optimizer.zero_grad()/step() — same
    contract as source_only.training_step, so train.py's loop body is
    identical regardless of which method is active.
    """
    source_images = batch["source_images"].to(device, non_blocking=True)
    source_labels = batch["source_labels"].to(device, non_blocking=True)
    target_images = batch["target_images"].to(device, non_blocking=True)

    features_s = model.backbone(source_images)
    features_t = model.backbone(target_images)

    logits_s = model.head(features_s)
    cls_loss = criterion(logits_s, source_labels)

    mmd_loss = mmd2(features_s, features_t, model.kernel_bandwidth_multipliers)

    total_loss = cls_loss + model.lambda_mmd * mmd_loss
    total_loss.backward()

    with torch.no_grad():
        preds = logits_s.argmax(dim=1)
        batch_acc = (preds == source_labels).float().mean().item()

    return {
        "loss": total_loss.item(),
        "cls_loss": cls_loss.item(),
        "mmd_loss": mmd_loss.item(),
        "batch_acc": batch_acc,
    }


def save_checkpoint(model: DANModel, path: str) -> None:
    """
    Same {"backbone": ..., "head": ...} format as source_only — DAN adds no
    extra learnable modules, so no extra checkpoint keys are needed.
    """
    torch.save(
        {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
        path,
    )


def load_checkpoint(model: DANModel, path: str, device: torch.device) -> DANModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model
