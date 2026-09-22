import torch.nn as nn
from task2.models.backbone import get_resnet18_backbone
from task2.models.classifier_head import PACSClassifier

class SourceOnly(nn.Module):
    """
    Baseline Empirical Risk Minimization (ERM).
    Trains exclusively on labeled source domains.
    """
    def __init__(self):
        super().__init__()
        self.backbone, feature_dim = get_resnet18_backbone()
        self.classifier = PACSClassifier(in_features=feature_dim)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features
        
    def compute_loss(self, src_images, src_labels, tgt_images=None):
        """
        Calculates the source-only cross-entropy loss.
        tgt_images is accepted for API consistency but not used.
        """
        logits, _ = self(src_images)
        loss = self.criterion(logits, src_labels)
        
        # Return loss and a dictionary of metrics for logging
        return loss, {"clf_loss": loss.item()}