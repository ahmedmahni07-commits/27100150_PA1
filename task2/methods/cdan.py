"""
Step 4: CDAN — class-conditional adversarial alignment. Same discriminator
shape/schedule/weight as DANN, but fed g(x) = vec(f ⊗ p) instead of the raw
feature. No entropy conditioning; features and probs are not detached.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifierHead
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    GradientReversalLayer,
    cdan_multilinear_map,
    grl_alpha_schedule,
    normalize_discriminator_input,
)


@dataclass
class CDANModel:
    backbone: ResNet18Backbone
    head: LinearClassifierHead
    discriminator: DomainDiscriminator
    grl: GradientReversalLayer
    domain_loss_weight: float
    grl_max_alpha: float

    def parameters(self):
        return (
            list(self.backbone.parameters())
            + list(self.head.parameters())
            + list(self.discriminator.parameters())
        )

    def train_mode(self) -> None:
        self.backbone.train()
        self.head.train()
        self.discriminator.train()
        self.backbone.freeze_batchnorm_running_stats()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()
        self.discriminator.eval()

    def to(self, device: torch.device) -> "CDANModel":
        self.backbone.to(device)
        self.head.to(device)
        self.discriminator.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> CDANModel:
    """
    Constructs backbone -> head -> discriminator, same order as DANN, so
    the classifier head's random init still matches source_only/DAN/DANN.
    Discriminator input dim is feature_dim * num_classes (the flattened
    outer product), not feature_dim alone as in DANN.
    """
    backbone = ResNet18Backbone(pretrained=True)
    num_classes = cfg["model"]["num_classes"]
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=num_classes,
    )
    discriminator = DomainDiscriminator(
        in_dim=backbone.feature_dim * num_classes,
        hidden_dim=cfg["cdan"]["discriminator_hidden_dim"],
        dropout=cfg["cdan"]["discriminator_dropout"],
    )
    grl = GradientReversalLayer(alpha=0.0)  # alpha is set per-step in training_step

    model = CDANModel(
        backbone=backbone,
        head=head,
        discriminator=discriminator,
        grl=grl,
        domain_loss_weight=cfg["cdan"]["domain_loss_weight"],
        grl_max_alpha=cfg["cdan"]["grl_max_alpha"],
    )
    model.to(device)
    return model


def training_step(
    model: CDANModel,
    batch: dict,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    progress_p: float = 0.0,
) -> dict:
    """
    Same overall shape as DANN's training_step, except the discriminator is
    fed g(x) = vec(f ⊗ p) (cdan_multilinear_map) instead of the raw feature.
    Per spec: no entropy conditioning, and features/probs are NOT detached
    before the multilinear map — gradients from the domain loss flow back
    into both the classifier head (via probs) and the backbone (via
    features), same as they do into the backbone alone in DANN.
    """
    model.grl.alpha = grl_alpha_schedule(progress_p, model.grl_max_alpha)

    source_images = batch["source_images"].to(device, non_blocking=True)
    source_labels = batch["source_labels"].to(device, non_blocking=True)
    target_images = batch["target_images"].to(device, non_blocking=True)

    features_s = model.backbone(source_images)
    features_t = model.backbone(target_images)

    logits_s = model.head(features_s)
    logits_t = model.head(features_t)
    cls_loss = criterion(logits_s, source_labels)

    probs_s = F.softmax(logits_s, dim=1)  # not detached
    probs_t = F.softmax(logits_t, dim=1)  # not detached

    g_s = cdan_multilinear_map(features_s, probs_s)
    g_t = cdan_multilinear_map(features_t, probs_t)
    domain_features = torch.cat([g_s, g_t], dim=0)
    domain_labels = torch.cat([
        torch.zeros(features_s.size(0), dtype=torch.long, device=device),
        torch.ones(features_t.size(0), dtype=torch.long, device=device),
    ])

    # DEVIATION from manual (see task2/DEVIATIONS.md): normalize only the
    # copy fed to the discriminator; cls_loss/probs above already used the
    # raw, un-normalized/un-detached features and probs, per spec.
    domain_features = normalize_discriminator_input(domain_features)
    reversed_features = model.grl(domain_features)
    domain_logits = model.discriminator(reversed_features)
    domain_loss = criterion(domain_logits, domain_labels)

    total_loss = cls_loss + model.domain_loss_weight * domain_loss
    total_loss.backward()

    with torch.no_grad():
        preds = logits_s.argmax(dim=1)
        batch_acc = (preds == source_labels).float().mean().item()
        domain_preds = domain_logits.argmax(dim=1)
        domain_acc = (domain_preds == domain_labels).float().mean().item()

    return {
        "loss": total_loss.item(),
        "cls_loss": cls_loss.item(),
        "domain_loss": domain_loss.item(),
        "batch_acc": batch_acc,
        "domain_acc": domain_acc,
        "grl_alpha": model.grl.alpha,
    }


def save_checkpoint(model: CDANModel, path: str) -> None:
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "head": model.head.state_dict(),
            "discriminator": model.discriminator.state_dict(),
        },
        path,
    )


def load_checkpoint(model: CDANModel, path: str, device: torch.device) -> CDANModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
    return model
