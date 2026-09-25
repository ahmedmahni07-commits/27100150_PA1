"""
Manifold mixup helpers for PROSER's data placeholders (Step 4): mixes the
post-layer2/pre-layer3 intermediate feature map (see
task4.models.resnet_cifar.ResNetCifarBackbone.forward_pre_layer3) between
two DIFFERENT-class examples, per Zhou et al. (2021)'s data-placeholder
construction.

Design choice (spec leaves this to us): lambda ~ Beta(2,2) is sampled ONE
VALUE PER PAIR, not one shared scalar for the whole half-batch -- this
gives a diverse spread of interpolation ratios across the batch's proxy
unknowns rather than one fixed mixing ratio every step, matching the
"provide examples of locations" framing (spec) better than a single
global lambda would.
"""

from __future__ import annotations

import torch


def sample_different_class_pairs(labels: torch.Tensor) -> torch.Tensor:
    """
    Given `labels` (N,), returns a permutation `perm` of range(N) such
    that labels[perm[i]] != labels[i] wherever possible. Drawn via a
    random permutation, with a bounded number of retries that reshuffle
    only the still-colliding positions among themselves. Any residual
    same-class collision after retries is NOT silently mixed -- the
    caller must combine this with valid_pair_mask() and only use pairs
    where the mask is True.
    """
    n = labels.size(0)
    device = labels.device
    perm = torch.randperm(n, device=device)
    for _ in range(10):
        collisions = labels[perm] == labels
        if not collisions.any():
            break
        idx = collisions.nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            perm[idx] = perm[idx[torch.randperm(idx.numel(), device=device)]]
        else:
            break  # a single leftover collision can't be fixed by permuting it alone
    return perm


def valid_pair_mask(labels: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Boolean mask: True where labels[perm[i]] != labels[i] (a genuine cross-class pair)."""
    return labels[perm] != labels


def sample_mixup_lambda(
    batch_size: int, device: torch.device, alpha: float = 2.0, beta: float = 2.0
) -> torch.Tensor:
    """One lambda ~ Beta(alpha, beta) per example/pair -- see module docstring."""
    # Sampled on CPU: aten::_sample_dirichlet (used by Beta) is not implemented
    # on MPS. Drawing N scalars on CPU and moving them is negligible cost and
    # keeps the global torch RNG stream deterministic under the seed.
    dist = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(beta)))
    return dist.sample((batch_size,)).to(device)


def mix_features(h_i: torch.Tensor, h_j: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    h~ = lambda*h_i + (1-lambda)*h_j, broadcasting lam (N,) over h_i/h_j's
    (N, C, H, W) post-layer2 feature maps.
    """
    lam = lam.view(-1, *([1] * (h_i.dim() - 1)))
    return lam * h_i + (1.0 - lam) * h_j
