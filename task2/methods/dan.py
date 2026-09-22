"""
DAN -- Maximum Mean Discrepancy (MMD) marginal alignment.

    L_DAN = L_cls + lambda_mmd * ||E_s[phi(F(x_s))] - E_t[phi(F(x_t))]||^2_H

MMD is estimated with the kernel trick (phi is never built explicitly) using
a sum of three RBF kernels, bandwidths = {0.5, 1, 2} x the median pairwise
squared feature distance in the current combined (source+target) batch.
This only pulls the *marginal* feature distributions together -- it never
looks at predicted class, which is exactly what distinguishes DAN from CDAN.
"""
import torch
import torch.nn as nn

from task2.models.backbone import get_resnet18_backbone
from task2.models.classifier_head import ClassifierHead

KERNEL_MULS = (0.5, 1.0, 2.0)


def _pairwise_sq_dists(x):
    """n x n squared-Euclidean-distance matrix via ||a-b||^2 = ||a||^2+||b||^2-2a.b,
    avoiding an explicit n x n x d expansion."""
    sq_norms = (x ** 2).sum(dim=1, keepdim=True)
    dists = sq_norms + sq_norms.t() - 2.0 * (x @ x.t())
    return dists.clamp(min=0.0)  # guards tiny negative values from floating-point error


def _median_bandwidth(dists):
    n = dists.size(0)
    off_diag = dists[~torch.eye(n, dtype=torch.bool, device=dists.device)]
    return off_diag.median()


def mmd2(source_feats, target_feats):
    """Biased multi-kernel MMD^2 estimate between two feature batches."""
    n_s = source_feats.size(0)
    combined = torch.cat([source_feats, target_feats], dim=0)
    dists = _pairwise_sq_dists(combined)
    # Bandwidth is a statistic of this batch's geometry, not a learned
    # parameter -- detach so no gradient flows through the median itself.
    bandwidth = _median_bandwidth(dists).detach().clamp(min=1e-6)

    kernel = sum(torch.exp(-dists / (bandwidth * mul)) for mul in KERNEL_MULS)

    k_ss = kernel[:n_s, :n_s]
    k_tt = kernel[n_s:, n_s:]
    k_st = kernel[:n_s, n_s:]
    return k_ss.mean() - 2.0 * k_st.mean() + k_tt.mean()


class DAN(nn.Module):
    def __init__(self, num_classes: int = 7, lambda_mmd: float = 1.0, **_ignored):
        super().__init__()
        self.backbone, feature_dim = get_resnet18_backbone()
        self.classifier = ClassifierHead(feature_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_mmd = lambda_mmd

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def compute_loss(self, src_images, src_labels, tgt_images, progress=0.0):
        src_logits, src_feats = self(src_images)
        _, tgt_feats = self(tgt_images)

        clf_loss = self.criterion(src_logits, src_labels)
        mmd_loss = mmd2(src_feats, tgt_feats)
        loss = clf_loss + self.lambda_mmd * mmd_loss

        return loss, {
            "clf_loss": clf_loss.item(),
            "mmd_loss": mmd_loss.item(),
            "total_loss": loss.item(),
        }
