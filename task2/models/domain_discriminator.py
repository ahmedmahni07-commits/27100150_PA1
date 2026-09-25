"""
Gradient-Reversal Layer + domain discriminator MLP, shared by DANN and CDAN.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Identity in the forward pass; multiplies the gradient by -alpha in the
    backward pass, implementing the gradient-reversal layer.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


def grl_alpha_schedule(p: float, max_alpha: float = 1.0) -> float:
    """
    alpha(p) = max_alpha * (2 / (1 + exp(-10p)) - 1), p in [0, 1]
    p = training progress, e.g. current_step / total_steps across ALL
    epochs (not just the current epoch) — define it that way in train.py
    so alpha ramps monotonically over the full run.

    max_alpha lets the controlled-design-study sweep {0.25, 0.5, 1} reuse
    this same schedule shape without changing its curve.
    """
    p = min(max(p, 0.0), 1.0)
    return max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def normalize_discriminator_input(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    DEVIATION from manual (see task2/DEVIATIONS.md): L2-normalizes each row
    before it reaches the domain discriminator -- for DANN the raw 512-d
    feature, for CDAN the flattened f (x) p vector -- so the discriminator's
    input scale (and therefore the domain loss/gradient magnitude the GRL
    reverses) is bounded regardless of how large the backbone's raw feature
    activations get. Applied ONLY on the domain-discriminator branch; the
    features/probs used for the classification loss and head are untouched.
    Added after observing train_domain_loss diverge past 1e8 within ~1 epoch
    with no such bound in place.
    """
    return F.normalize(x, p=2, dim=1, eps=eps)


class DomainDiscriminator(nn.Module):
    """
    256-unit hidden layer, ReLU, dropout 0.5, 2-class output.
    Input dim differs between DANN (512, the raw feature) and CDAN
    (512 * num_classes, the vec(f ⊗ p) outer product) — pass in_dim
    explicitly from the calling method.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cdan_multilinear_map(features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """
    g(x) = vec(f ⊗ p): outer product of the 512-d feature and the C-way
    softmax probability vector, flattened to (512 * C,) per example.

    features: (B, D)
    probs:    (B, C)   -- softmax(logits), NOT detached (per spec: no
                           entropy conditioning, do not detach f or p)
    returns:  (B, D*C)
    """
    batch_size = features.size(0)
    outer = torch.bmm(features.unsqueeze(2), probs.unsqueeze(1))  # (B, D, C)
    return outer.view(batch_size, -1)