"""
DANN -- adversarial domain alignment via a gradient-reversal layer (GRL).

A domain discriminator is attached to the 512-d feature through the GRL: the
discriminator itself is trained normally to tell source from target, but the
GRL negates (and schedules) the gradient it sends back into the backbone, so
the backbone is simultaneously pushed to make the two domains indistinguish-
able. Only source examples contribute to the classification loss; both
source and target contribute to the domain loss (unit weight, per spec).
"""
import torch
import torch.nn as nn

from task2.models.backbone import get_resnet18_backbone
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator, gradient_reverse, grl_alpha


class DANN(nn.Module):
    def __init__(self, num_classes: int = 7, lambda_dom: float = 1.0, **_ignored):
        super().__init__()
        self.backbone, feature_dim = get_resnet18_backbone()
        self.classifier = ClassifierHead(feature_dim, num_classes)
        self.discriminator = DomainDiscriminator(feature_dim)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.CrossEntropyLoss()
        self.lambda_dom = lambda_dom

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def compute_loss(self, src_images, src_labels, tgt_images, progress=0.0):
        src_logits, src_feats = self(src_images)
        _, tgt_feats = self(tgt_images)

        clf_loss = self.cls_criterion(src_logits, src_labels)

        alpha = grl_alpha(progress)  # progress in [0,1]: current_step / total_steps, set by train.py
        domain_feats = torch.cat([src_feats, tgt_feats], dim=0)
        domain_logits = self.discriminator(gradient_reverse(domain_feats, alpha))

        n_s, n_t = src_feats.size(0), tgt_feats.size(0)
        domain_labels = torch.cat([
            torch.zeros(n_s, dtype=torch.long, device=domain_logits.device),  # 0 = source
            torch.ones(n_t, dtype=torch.long, device=domain_logits.device),   # 1 = target
        ])
        dom_loss = self.dom_criterion(domain_logits, domain_labels)

        loss = clf_loss + self.lambda_dom * dom_loss
        with torch.no_grad():
            dom_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()

        return loss, {
            "clf_loss": clf_loss.item(),
            "dom_loss": dom_loss.item(),
            "dom_acc": dom_acc,   # near 0.5 = discriminator confused = domains well-aligned
            "alpha": alpha,
            "total_loss": loss.item(),
        }
