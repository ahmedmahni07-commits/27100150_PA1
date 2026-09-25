"""
Local sharpness proxy diagnostic (Required Evidence for the ERM vs. SAM
comparison): on a FIXED batch (32 examples per source domain = 96 total,
drawn once with seed 6304 and reused unchanged for every method so the
comparison is apples-to-apples), take one normalized gradient-ascent
perturbation of magnitude rho and report

    Delta_sharp = L(theta + eps) - L(theta)

A flatter minimum -- what SAM training is meant to find -- should show a
smaller Delta_sharp than plain ERM's.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common.pacs_protocol import DomainSplit, SEED


def build_fixed_sharpness_batch(
    source_splits: dict,
    eval_transform,
    device: torch.device,
    per_domain: int = 32,
    seed: int = SEED,
):
    """
    Draws exactly `per_domain` examples per source domain from each
    domain's VALIDATION split (per spec: "a fixed validation batch"),
    deterministically from `seed`. Callers should build this ONCE and
    reuse the identical tensors across every method being compared
    (ERM / DAN-DG / SAM), so any Delta_sharp difference reflects the loss
    landscape at each method's optimum, not batch noise.

    Returns (images, labels) tensors already moved to `device`.
    """
    rng = np.random.RandomState(seed)
    images, labels = [], []
    for domain in sorted(source_splits.keys()):
        split: DomainSplit = source_splits[domain]
        n = len(split.val_paths)
        idx = rng.choice(n, size=min(per_domain, n), replace=False)
        for i in idx:
            img = Image.open(split.val_paths[i]).convert("RGB")
            images.append(eval_transform(img))
            labels.append(split.val_labels[i])

    images_t = torch.stack(images).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long).to(device)
    return images_t, labels_t


def compute_local_sharpness(
    model,
    fixed_batch: tuple,
    criterion: nn.CrossEntropyLoss,
    rho: float = 0.05,
) -> float:
    """
    theta -> compute L(theta) and its gradient -> one normalized ascent
    step of size rho -> compute L(theta+eps) -> restore theta exactly ->
    return the gap. Per spec the model is placed in EVALUATION mode
    (model.eval_mode()) before computing both losses -- gradients still
    flow through eval-mode BatchNorm/head just fine, this only changes
    which BN statistics get used. Leaves the model's parameters
    bit-for-bit unchanged and clears .grad afterward.

    Works with any of ERMModel / DANDGModel / SAMModel: all three expose
    the same .backbone / .head / .eval_mode() / .parameters() interface.
    """
    images, labels = fixed_batch
    params = model.parameters()

    model.eval_mode()
    for p in params:
        p.grad = None

    features = model.backbone(images)
    logits = model.head(features)
    loss_theta = criterion(logits, labels)
    loss_theta.backward()

    grads = [p.grad for p in params if p.grad is not None]
    grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2)
    scale = rho / (grad_norm + 1e-12)

    eps_list = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                eps_list.append(None)
                continue
            eps = p.grad * scale
            p.add_(eps)
            eps_list.append(eps)

    with torch.no_grad():
        features_pert = model.backbone(images)
        logits_pert = model.head(features_pert)
        loss_pert = criterion(logits_pert, labels)

    with torch.no_grad():
        for p, eps in zip(params, eps_list):
            if eps is not None:
                p.sub_(eps)
    for p in params:
        p.grad = None

    return float(loss_pert.item() - loss_theta.item())
