"""
Step: SAM (Sharpness-Aware Minimization), applied to the plain ERM
classification loss over domain-balanced SOURCE batches (no target, no MMD
term -- SAM's mechanism is purely about the shape of the loss landscape it
converges to, not explicit domain alignment).

SAM solves min_theta max_{||eps||_2 <= rho} L(theta + eps) via the
standard two-pass approximation (Foret et al. 2021):
  1. forward/backward at theta -> gradient g
  2. ascent step: eps = rho * g / (||g||_2 + 1e-12); theta <- theta + eps
  3. forward/backward AGAIN at theta+eps, on the SAME batch -> gradient g'
  4. restore theta <- theta - eps, then apply the base optimizer's real
     update using g'

This needs two forward/backward passes per step, which doesn't fit the
uniform "zero_grad(); training_step(); optimizer.step()" loop that ERM/
DAN-DG use -- task3/train.py special-cases method == "sam" with its own
loop built from SAMOptimizer + training_step_first_pass +
training_step_second_pass below.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import LinearClassifierHead


@dataclass
class SAMModel:
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

    def to(self, device: torch.device) -> "SAMModel":
        self.backbone.to(device)
        self.head.to(device)
        return self


def build_model(cfg: dict, device: torch.device) -> SAMModel:
    backbone = ResNet18Backbone(pretrained=True)
    head = LinearClassifierHead(
        feature_dim=backbone.feature_dim,
        num_classes=cfg["model"]["num_classes"],
    )
    model = SAMModel(backbone=backbone, head=head)
    model.to(device)
    return model


class SAMOptimizer:
    """
    Minimal SAM wrapper around a base optimizer (AdamW). Holds its own flat
    `params` list -- matching this repo's Model.parameters() convention
    (a plain list, not PyTorch's param_groups) -- since that's all
    task3/train.py's uniform construction needs.

    Usage per step:
        optimizer.zero_grad()
        <forward/backward at theta, populating p.grad>
        optimizer.ascent_step()      # theta -> theta + eps
        optimizer.zero_grad()
        <forward/backward AGAIN at theta+eps, on the same batch>
        optimizer.descent_step()     # theta+eps -> theta, then real update
    """

    def __init__(
        self,
        params,
        base_optimizer_cls=torch.optim.AdamW,
        rho: float = 0.05,
        **base_optimizer_kwargs,
    ):
        self.params = list(params)
        self.rho = rho
        self.base_optimizer = base_optimizer_cls(self.params, **base_optimizer_kwargs)
        self._eps: list = []

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def ascent_step(self) -> None:
        """After the FIRST backward() (at theta): perturb theta -> theta + eps."""
        grads = [p.grad for p in self.params if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2)
        scale = self.rho / (grad_norm + 1e-12)

        self._eps = []
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    self._eps.append(None)
                    continue
                eps = p.grad * scale
                p.add_(eps)
                self._eps.append(eps)

    def descent_step(self) -> None:
        """
        After the SECOND backward() (at theta+eps, so p.grad now holds the
        perturbed-point gradient): restore theta+eps -> theta, then apply
        the real AdamW update using that gradient.
        """
        with torch.no_grad():
            for p, eps in zip(self.params, self._eps):
                if eps is not None:
                    p.sub_(eps)
        self.base_optimizer.step()
        self._eps = []


def _forward_loss(model: SAMModel, batch: dict, criterion: nn.CrossEntropyLoss, device: torch.device):
    """Shared forward pass, used identically by both SAM passes on the same batch."""
    images = batch["source_images"].to(device, non_blocking=True)
    labels = batch["source_labels"].to(device, non_blocking=True)
    features = model.backbone(images)
    logits = model.head(features)
    loss = criterion(logits, labels)
    return loss, logits, labels


def training_step_first_pass(
    model: SAMModel, batch: dict, criterion: nn.CrossEntropyLoss, device: torch.device
) -> dict:
    """First forward/backward at theta -- its gradient drives SAMOptimizer.ascent_step()."""
    loss, logits, labels = _forward_loss(model, batch, criterion, device)
    loss.backward()
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()
    return {"first_pass_loss": loss.item(), "batch_acc": batch_acc}


def training_step_second_pass(
    model: SAMModel, batch: dict, criterion: nn.CrossEntropyLoss, device: torch.device
) -> dict:
    """Second forward/backward at theta+eps -- its gradient drives SAMOptimizer.descent_step()."""
    loss, _logits, _labels = _forward_loss(model, batch, criterion, device)
    loss.backward()
    return {"loss": loss.item(), "cls_loss": loss.item()}


def save_checkpoint(model: SAMModel, path: str) -> None:
    torch.save(
        {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
        path,
    )


def load_checkpoint(model: SAMModel, path: str, device: torch.device) -> SAMModel:
    state = torch.load(path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])
    return model
