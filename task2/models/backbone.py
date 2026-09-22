"""
ResNet-18 backbone for Task 2/3, with the assignment's BatchNorm policy:
running mean/var stay frozen at their pretrained ImageNet values for every
method; only the BN affine parameters (gamma/beta) remain trainable.
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def get_resnet18_backbone():
    """Loads ResNet-18 (IMAGENET1K_V1), strips the classification head.
    Returns (backbone, feature_dim=512)."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


def freeze_bn_running_stats(module):
    """Call as model.apply(freeze_bn_running_stats) AFTER model.train().

    Puts every BatchNorm submodule back into eval() so its running_mean/
    running_var stop updating, while everything else stays in training
    mode. Must be re-applied every time .train() is called, since .train()
    recursively re-enables BN's running-stat updates on every submodule.
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()
