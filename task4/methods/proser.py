"""
Step 4: PROSER -- classifier + data placeholders for open-set rejection
(Zhou et al. 2021, "Learning Placeholders for Open-Set Recognition").
Initialized from the selected Vanilla checkpoint; appends
num_dummy_classifiers randomly initialized dummy classifiers and
fine-tunes the WHOLE model (backbone + known-class head + dummy head)
toward a two-part objective.

Per the paper's Eq. 5/7 (confirmed against arXiv 2103.15086): with C
dummy classifiers, their logits are collapsed into ONE placeholder
response via max-pooling, forming a (K+1)-way vector
    full_logits = cat([known_logits (K), max_c dummy_logits_c], dim=-1)

  Classifier placeholders (first half of each batch, ordinary examples):
    l1 = CE(full_logits, y) + beta * CE(full_logits_with_y_masked, K)
  "Masking y" (the paper's "set the ground-truth probability to 0 and
  renormalize") is implemented by setting that one logit to -inf before
  softmax/cross-entropy -- exactly equivalent, since the -inf'd term
  drops out of the softmax denominator, and simpler to batch/vectorize.

  Data placeholders (second half of each batch, manifold mixup):
    x_i, x_j drawn from DIFFERENT known classes (manifold_mixup.py) ->
    h_i, h_j = backbone.forward_pre_layer3(x_i), (x_j)  # after layer2, before layer3
    h~ = lambda*h_i + (1-lambda)*h_j,  lambda ~ Beta(2,2) per pair
    z~ = backbone.forward_post_layer3(h~); full_logits~ = cat([head(z~), max_c dummy_c(z~)])
    l2 = CE(full_logits~, K)   # no masking -- neither original class is correct

  Total per-step loss = l1 + gamma * l2, ONE backward() call -- same
  training_step(model, batch, criterion, device) -> metrics contract as
  vanilla/gcsc, so task4/train.py's generic loop needs no PROSER-specific
  branching beyond method dispatch; PROSER's different optimizer/epoch
  budget and vanilla-checkpoint initialization live entirely in this
  module + task4/configs/proser.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from task4.methods import manifold_mixup, vanilla
from task4.models.classifier_head import LinearClassifierHead
from task4.models.resnet_cifar import ResNetCifarBackbone


@dataclass
class PROSERModel:
    backbone: ResNetCifarBackbone
    head: LinearClassifierHead  # 10 known-class logits
    dummy_head: nn.Linear  # num_dummy_classifiers extra logits
    beta: float
    gamma: float
    mixup_alpha: float
    mixup_beta: float

    def parameters(self):
        return (
            list(self.backbone.parameters())
            + list(self.head.parameters())
            + list(self.dummy_head.parameters())
        )

    def train_mode(self) -> None:
        self.backbone.train()
        self.head.train()
        self.dummy_head.train()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()
        self.dummy_head.eval()

    def to(self, device: torch.device) -> "PROSERModel":
        self.backbone.to(device)
        self.head.to(device)
        self.dummy_head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> PROSERModel:
    backbone = ResNetCifarBackbone()
    head = LinearClassifierHead(feature_dim=backbone.feature_dim, num_classes=cfg["data"]["num_classes"])
    dummy_head = nn.Linear(backbone.feature_dim, cfg["proser"]["num_dummy_classifiers"])

    model = PROSERModel(
        backbone=backbone,
        head=head,
        dummy_head=dummy_head,
        beta=cfg["proser"]["beta"],
        gamma=cfg["proser"]["gamma"],
        mixup_alpha=cfg["proser"]["mixup_beta_a"],
        mixup_beta=cfg["proser"]["mixup_beta_b"],
    )
    model.to(device)

    # Initialize backbone + known-class head from the selected Vanilla
    # checkpoint (spec: "Initialize PROSER from the selected Vanilla
    # checkpoint"); only dummy_head starts from random init.
    vanilla_shell = vanilla.VanillaModel(backbone=model.backbone, head=model.head)
    vanilla.load_checkpoint(vanilla_shell, cfg["proser"]["vanilla_checkpoint"], device)

    return model


def _collapsed_logits(known_logits: torch.Tensor, dummy_logits: torch.Tensor) -> torch.Tensor:
    """cat([known_logits (N,K), max-over-dummies (N,1)], dim=-1) -> (N, K+1)."""
    placeholder = dummy_logits.max(dim=-1, keepdim=True).values
    return torch.cat([known_logits, placeholder], dim=-1)


def training_step(
    model: PROSERModel,
    batch,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    n = images.size(0)
    n_first = n // 2
    first_images, first_labels = images[:n_first], labels[:n_first]
    second_images, second_labels = images[n_first:], labels[n_first:]

    num_known = model.head.fc.out_features
    placeholder_idx = num_known  # last position in the (K+1)-way collapsed vector

    # --- Classifier placeholders (first half, ordinary examples) ---
    features_1 = model.backbone(first_images)
    known_logits_1 = model.head(features_1)
    dummy_logits_1 = model.dummy_head(features_1)
    full_logits_1 = _collapsed_logits(known_logits_1, dummy_logits_1)

    l1_main = criterion(full_logits_1, first_labels)

    masked_logits_1 = full_logits_1.clone()
    masked_logits_1.scatter_(1, first_labels.unsqueeze(1), float("-inf"))
    placeholder_targets_1 = torch.full_like(first_labels, fill_value=placeholder_idx)
    l1_mask = criterion(masked_logits_1, placeholder_targets_1)

    l1 = l1_main + model.beta * l1_mask

    # --- Data placeholders (second half, manifold mixup) ---
    l2 = torch.zeros((), device=device)
    if second_images.size(0) >= 2:
        perm = manifold_mixup.sample_different_class_pairs(second_labels)
        valid = manifold_mixup.valid_pair_mask(second_labels, perm)

        if valid.any():
            h_i = model.backbone.forward_pre_layer3(second_images)
            h_j = h_i[perm]
            lam = manifold_mixup.sample_mixup_lambda(
                second_images.size(0), device, model.mixup_alpha, model.mixup_beta
            )
            h_mix = manifold_mixup.mix_features(h_i, h_j, lam)

            features_2 = model.backbone.forward_post_layer3(h_mix)
            known_logits_2 = model.head(features_2)
            dummy_logits_2 = model.dummy_head(features_2)
            full_logits_2 = _collapsed_logits(known_logits_2, dummy_logits_2)

            placeholder_targets_2 = torch.full(
                (full_logits_2.size(0),), placeholder_idx, dtype=torch.long, device=device
            )
            l2 = criterion(full_logits_2[valid], placeholder_targets_2[valid])

    total_loss = l1 + model.gamma * l2
    total_loss.backward()

    with torch.no_grad():
        preds_1 = known_logits_1.argmax(dim=1)
        batch_acc = (preds_1 == first_labels).float().mean().item()

    return {
        "loss": total_loss.item(),
        "l1_classifier_placeholder": l1.item(),
        "l2_data_placeholder": l2.item(),
        "batch_acc": batch_acc,
    }


def save_checkpoint(model: PROSERModel, path: str) -> None:
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "head": model.head.state_dict(),
            "dummy_head": model.dummy_head.state_dict(),
        },
        path,
    )


def load_checkpoint(model: PROSERModel, path: str, device: torch.device) -> PROSERModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    model.dummy_head.load_state_dict(state["dummy_head"])
    return model


def placeholder_score(model: PROSERModel, features: torch.Tensor) -> torch.Tensor:
    """
    PROSER's own placeholder-based unknownness score: softmax probability
    mass on the collapsed placeholder position, over the (K+1)-way vector
    [known_logits, max_c dummy_logits_c]. Larger = more novel, same u(x)
    convention as MSP/MLS/Energy/Mahalanobis. (task4.evaluation.metrics.
    compute_proser_placeholder_score is the numpy mirror of this, used by
    evaluate_osr.py against extract_outputs.py's cached arrays.)
    """
    known_logits = model.head(features)
    dummy_logits = model.dummy_head(features)
    full_logits = _collapsed_logits(known_logits, dummy_logits)
    probs = torch.softmax(full_logits, dim=-1)
    return probs[:, -1]
