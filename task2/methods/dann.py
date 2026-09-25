"""
Step 3: DANN — adversarial domain alignment via a gradient-reversal layer
and a binary domain discriminator on the 512-d feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifierHead
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    GradientReversalLayer,
    grl_alpha_schedule,
    normalize_discriminator_input,
)


@dataclass
class DANNModel:
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

    def to(self, device: torch.device) -> "DANNModel":
        self.backbone.to(device)
        self.head.to(device)
        self.discriminator.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> DANNModel:
    """
    Constructs backbone -> head -> discriminator, in that order, so the
    classifier head's random init matches source_only/DAN exactly (the
    head is built before the discriminator, so the discriminator's own
    random init draws RNG state that comes strictly after the head's).
    """
    backbone = ResNet18Backbone(pretrained=True)
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=cfg["model"]["num_classes"],
    )
    discriminator = DomainDiscriminator(
        in_dim=backbone.feature_dim,
        hidden_dim=cfg["dann"]["discriminator_hidden_dim"],
        dropout=cfg["dann"]["discriminator_dropout"],
    )
    grl = GradientReversalLayer(alpha=0.0)  # alpha is set per-step in training_step

    model = DANNModel(
        backbone=backbone,
        head=head,
        discriminator=discriminator,
        grl=grl,
        domain_loss_weight=cfg["dann"]["domain_loss_weight"],
        grl_max_alpha=cfg["dann"]["grl_max_alpha"],
    )
    model.to(device)
    return model


def training_step(
    model: DANNModel,
    batch: dict,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    progress_p: float = 0.0,
) -> dict:
    """
    `progress_p` = current global training step / total planned training
    steps across the ENTIRE run (all epochs, not just the current one),
    in [0, 1] — computed and passed in by train.py. The single `criterion`
    (a plain, stateless CrossEntropyLoss) is reused for both the class loss
    and the domain loss.

    - alpha = grl_alpha_schedule(progress_p, model.grl_max_alpha); set on model.grl
    - cls_loss: source only, CE against source labels
    - domain_loss: BOTH source (domain label 0) and target (domain label 1)
      features, passed through the GRL then the discriminator, CE against
      domain labels
    - total = cls_loss + model.domain_loss_weight * domain_loss

    Calls loss.backward() but not optimizer.zero_grad()/step() — same
    contract as source_only/dan, so train.py's loop body doesn't change.
    """
    model.grl.alpha = grl_alpha_schedule(progress_p, model.grl_max_alpha)

    source_images = batch["source_images"].to(device, non_blocking=True)
    source_labels = batch["source_labels"].to(device, non_blocking=True)
    target_images = batch["target_images"].to(device, non_blocking=True)

    features_s = model.backbone(source_images)
    features_t = model.backbone(target_images)

    logits_s = model.head(features_s)
    cls_loss = criterion(logits_s, source_labels)

    domain_features = torch.cat([features_s, features_t], dim=0)
    domain_labels = torch.cat([
        torch.zeros(features_s.size(0), dtype=torch.long, device=device),
        torch.ones(features_t.size(0), dtype=torch.long, device=device),
    ])

    # DEVIATION from manual (see task2/DEVIATIONS.md): normalize only the
    # copy of the features fed to the discriminator; cls_loss above already
    # used the raw, un-normalized features_s.
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


def save_checkpoint(model: DANNModel, path: str) -> None:
    """
    Carries backbone + head + discriminator (unlike source_only/DAN's
    backbone+head-only format) — evaluate_final.py only needs backbone+head,
    but keeping the discriminator costs little and lets you inspect/resume
    the adversarial half directly if needed.
    """
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "head": model.head.state_dict(),
            "discriminator": model.discriminator.state_dict(),
        },
        path,
    )


def load_checkpoint(model: DANNModel, path: str, device: torch.device) -> DANNModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
    return model
