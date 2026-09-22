"""
Domain discriminator + gradient-reversal layer, shared by DANN and CDAN.

Both methods use the SAME discriminator architecture and the SAME
gradient-reversal schedule alpha(p) = 2/(1+exp(-10p)) - 1; they differ only
in what feature is fed in (DANN: f; CDAN: vec(f (x) p), built by the caller).
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)


def grl_alpha(p: float) -> float:
    """p in [0,1] = training progress (current_step / total_steps)."""
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, 2-class output."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)
