"""
Step 5 diagnostic: can a simple classifier tell source features from target
features? Freeze the backbone, collect equal numbers of source-val and
target features, 70/30 split (seed 6304), balanced logistic regression
(C=1). Held-out accuracy = domain separability score (50% = chance).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def compute_domain_separability(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int = 6304,
    test_size: float = 0.3,
) -> float:
    """
    - subsample the larger of source_features/target_features so counts
      are equal ("equal numbers of source-validation and target features")
    - build X = concat(source, target), y = concat(zeros, ones)
    - 70/30 train/test split, stratified by domain label, seed=seed
    - LogisticRegression(C=1, class_weight="balanced")
    - return held-out accuracy
    """
    rng = np.random.RandomState(seed)
    n = min(len(source_features), len(target_features))

    if len(source_features) > n:
        idx = rng.choice(len(source_features), size=n, replace=False)
        source_features = source_features[idx]
    if len(target_features) > n:
        idx = rng.choice(len(target_features), size=n, replace=False)
        target_features = target_features[idx]

    X = np.concatenate([source_features, target_features], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))
