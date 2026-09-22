"""
CDAN -- class-conditional adversarial alignment.

Identical discriminator architecture, GRL, alpha(p) schedule, and loss
weight to DANN, but the discriminator is fed
    g(x) = vec(f (x) p)
the flattened outer product of the 512-d feature f and the classifier's
softmax probability vector p = C(f), instead of f alone -- so alignment is
conditioned on the classifier's current belief about each example's class.
Per the assignment's required implementation: no entropy conditioning, and
neither f nor p is detached (gradients from the domain loss flow through
both the backbone and the classifier head via this term).

NOTE: the assignment text says to reuse "the same hidden width, activation,
dropout, gradient-reversal schedule, and loss weight used for DAN" -- DAN
itself has no discriminator, so this is read as DANN (the only method with a
discriminator to reuse settings from). State this reading explicitly in the
report if asked to justify it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.models.backbone import get_resnet18_backbone
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator, gradient_reverse, grl_alpha


class CDAN(nn.Module):
    def __init__(self, num_classes: int = 7, lambda_dom: float = 1.0, **_ignored):
        super().__init__()
        self.backbone, feature_dim = get_resnet18_backbone()
        self.classifier = ClassifierHead(feature_dim, num_classes)
        # Discriminator input is the flattened f (x) p outer product: d * C.
        self.discriminator = DomainDiscriminator(feature_dim * num_classes)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.CrossEntropyLoss()
        self.lambda_dom = lambda_dom

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    @staticmethod
    def _multilinear_map(features, probs):
        """f (x) p, flattened: (B, d) outer (B, C) -> (B, d*C)."""
        b, d = features.shape
        c = probs.shape[1]
        return torch.bmm(features.unsqueeze(2), probs.unsqueeze(1)).view(b, d * c)

    def compute_loss(self, src_images, src_labels, tgt_images, progress=0.0):
        src_logits, src_feats = self(src_images)
        tgt_logits, tgt_feats = self(tgt_images)

        clf_loss = self.cls_criterion(src_logits, src_labels)

        # Neither f nor p is detached here, per spec.
        src_probs = F.softmax(src_logits, dim=1)
        tgt_probs = F.softmax(tgt_logits, dim=1)
        src_g = self._multilinear_map(src_feats, src_probs)
        tgt_g = self._multilinear_map(tgt_feats, tgt_probs)

        alpha = grl_alpha(progress)
        domain_input = torch.cat([src_g, tgt_g], dim=0)
        domain_logits = self.discriminator(gradient_reverse(domain_input, alpha))

        n_s, n_t = src_feats.size(0), tgt_feats.size(0)
        domain_labels = torch.cat([
            torch.zeros(n_s, dtype=torch.long, device=domain_logits.device),
            torch.ones(n_t, dtype=torch.long, device=domain_logits.device),
        ])
        dom_loss = self.dom_criterion(domain_logits, domain_labels)

        loss = clf_loss + self.lambda_dom * dom_loss
        with torch.no_grad():
            dom_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()

        return loss, {
            "clf_loss": clf_loss.item(),
            "dom_loss": dom_loss.item(),
            "dom_acc": dom_acc,
            "alpha": alpha,
            "total_loss": loss.item(),
        }
