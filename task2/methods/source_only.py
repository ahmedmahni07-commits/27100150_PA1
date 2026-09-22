"""
Source-only ERM baseline: ordinary cross-entropy on the three labeled source
domains. Never touches target images -- establishes the unadapted domain
gap that DAN/DANN/CDAN are compared against, and is reused unchanged as the
Task 3 ERM baseline.
"""
import torch.nn as nn

from task2.models.backbone import get_resnet18_backbone
from task2.models.classifier_head import ClassifierHead


class SourceOnly(nn.Module):
    def __init__(self, num_classes: int = 7, **_ignored):
        super().__init__()
        self.backbone, feature_dim = get_resnet18_backbone()
        self.classifier = ClassifierHead(feature_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def compute_loss(self, src_images, src_labels, tgt_images=None, progress=0.0):
        """Unified per-method API (also implemented by DAN/DANN/CDAN) so
        train.py can call any method the same way."""
        logits, _ = self(src_images)
        loss = self.criterion(logits, src_labels)
        return loss, {"clf_loss": loss.item()}
