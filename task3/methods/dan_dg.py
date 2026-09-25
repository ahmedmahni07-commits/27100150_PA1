"""
Step: DAN-DG -- domain-generalization analogue of Task 2's DAN. There is no
target domain available here (Sketch is never loaded during Task 3
training), so instead of aligning source vs. target features, this aligns
the three SOURCE domains' features with EACH OTHER, on the theory that a
representation invariant across photo/art/cartoon is more likely to also
transfer to the unseen sketch domain at test time.

L_DAN-DG = L_cls(all source) + lambda_dg * mean_{domain pairs} MMD^2(f_i, f_j)
over the C(3,2)=3 source-domain pairs (photo-art, photo-cartoon, art-cartoon).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn as nn

from common.mmd import mmd2
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import LinearClassifierHead


@dataclass
class DANDGModel:
    backbone: ResNet18Backbone
    head: LinearClassifierHead
    lambda_dg: float
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

    def to(self, device: torch.device) -> "DANDGModel":
        self.backbone.to(device)
        self.head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> DANDGModel:
    backbone = ResNet18Backbone(pretrained=True)
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=cfg["model"]["num_classes"],
    )
    model = DANDGModel(
        backbone=backbone,
        head=head,
        lambda_dg=cfg["dan_dg"]["lambda_dg"],
        kernel_bandwidth_multipliers=tuple(cfg["dan_dg"]["kernel_bandwidth_multipliers"]),
    )
    model.to(device)
    return model


def training_step(
    model: DANDGModel,
    batch: dict,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    progress_p: float = 0.0,  # unused; present so task3/train.py's call is uniform across methods
) -> dict:
    """
    `batch` is what common.pacs_protocol.source_balanced_batches yields:
    {"source_images", "source_labels", "source_domain_ids", "domain_names"}
    -- note there is no "target_images" key; Sketch is never in this batch.

    - forward all source_images -> features -> logits -> cls_loss (all source labels)
    - split features by source_domain_ids into the 3 per-domain groups
    - mmd_loss = mean over the 3 unordered domain pairs of mmd2(f_i, f_j)
    - total = cls_loss + model.lambda_dg * mmd_loss

    Calls loss.backward() but not optimizer.zero_grad()/step() -- same
    contract as task2's DAN, so task3/train.py's standard loop body is
    identical for every non-SAM method.
    """
    images = batch["source_images"].to(device, non_blocking=True)
    labels = batch["source_labels"].to(device, non_blocking=True)
    domain_ids = batch["source_domain_ids"].to(device, non_blocking=True)

    features = model.backbone(images)
    logits = model.head(features)
    cls_loss = criterion(logits, labels)

    num_domains = len(batch["domain_names"])
    per_domain_features = [features[domain_ids == d] for d in range(num_domains)]

    pair_losses = [
        mmd2(per_domain_features[i], per_domain_features[j], model.kernel_bandwidth_multipliers)
        for i, j in combinations(range(num_domains), 2)
    ]
    mmd_loss = torch.stack(pair_losses).mean()

    total_loss = cls_loss + model.lambda_dg * mmd_loss
    total_loss.backward()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()

    return {
        "loss": total_loss.item(),
        "cls_loss": cls_loss.item(),
        "mmd_loss": mmd_loss.item(),
        "batch_acc": batch_acc,
    }


def save_checkpoint(model: DANDGModel, path: str) -> None:
    torch.save(
        {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
        path,
    )


def load_checkpoint(model: DANDGModel, path: str, device: torch.device) -> DANDGModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model
