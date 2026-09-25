"""
u_Mah(x) = min_c (f(x)-mu_c)^T Sigma^-1 (f(x)-mu_c): Mahalanobis distance
to the nearest class-conditional Gaussian. Per spec: estimate class means
mu_c and ONE shared covariance Sigma from UNAUGMENTED CIFAR-10 TRAINING
features (never CIFAR-100), adding 1e-6 to every diagonal entry of Sigma.
"""

from __future__ import annotations

import numpy as np


def fit_class_gaussians(
    train_features: np.ndarray, train_labels: np.ndarray, num_classes: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (class_means (K,D), shared_precision (D,D)) -- ONE pooled
    covariance shared across all classes (the standard Mahalanobis-OOD
    construction, e.g. Lee et al. 2018), not per-class covariances.
    """
    feature_dim = train_features.shape[1]
    class_means = np.zeros((num_classes, feature_dim), dtype=np.float64)
    centered_chunks = []
    for c in range(num_classes):
        class_feats = train_features[train_labels == c].astype(np.float64)
        mu_c = class_feats.mean(axis=0)
        class_means[c] = mu_c
        centered_chunks.append(class_feats - mu_c)

    centered = np.concatenate(centered_chunks, axis=0)
    n = centered.shape[0]
    shared_cov = (centered.T @ centered) / n
    shared_cov = shared_cov + 1e-6 * np.eye(feature_dim)
    shared_precision = np.linalg.inv(shared_cov)
    return class_means, shared_precision


def compute_mahalanobis(
    features: np.ndarray, class_means: np.ndarray, shared_precision: np.ndarray
) -> np.ndarray:
    """features: (N, D). Returns u_Mah: (N,) = min_c squared Mahalanobis distance to class c."""
    num_classes = class_means.shape[0]
    dists = np.zeros((features.shape[0], num_classes), dtype=np.float64)
    features = features.astype(np.float64)
    for c in range(num_classes):
        diff = features - class_means[c]
        dists[:, c] = np.einsum("nd,de,ne->n", diff, shared_precision, diff)
    return dists.min(axis=1)
