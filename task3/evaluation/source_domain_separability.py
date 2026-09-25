"""
Task 3 diagnostic: source-domain separability. Can a simple classifier
tell which of the three SOURCE domains (photo / art_painting / cartoon) a
frozen-backbone feature came from? Chance = 1/3 (33.3%). This is the
Task 3 analogue of Task 2's source-vs-target domain_separability.py, but
3-way among the sources instead of binary source-vs-target -- Sketch is
never involved in this diagnostic.

Same recipe as Task 2's version: equal-count balancing across the classes
being separated, 70/30 stratified split (seed 6304), balanced
LogisticRegression(C=1).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def compute_source_domain_separability(
    features_by_domain: dict,
    seed: int = 6304,
    test_size: float = 0.3,
) -> dict:
    """
    features_by_domain: {"photo": np.ndarray[N_p, D], "art_painting": ...,
    "cartoon": ...} -- frozen-backbone features from each source domain's
    validation split.

    - subsample every domain down to the smallest domain's count (equal
      numbers per class, same balancing rule as Task 2's diagnostic)
    - X = concat over domains, y = domain index (0/1/2, by sorted domain name)
    - 70/30 stratified split, LogisticRegression(C=1, class_weight="balanced")
    - returns held-out accuracy (chance = 1/3) plus the confusion matrix,
      so you can see WHICH domain pairs are most/least separable
    """
    rng = np.random.RandomState(seed)
    domains = sorted(features_by_domain.keys())
    n = min(len(features_by_domain[d]) for d in domains)

    X_parts, y_parts = [], []
    for idx, d in enumerate(domains):
        feats = features_by_domain[d]
        if len(feats) > n:
            sel = rng.choice(len(feats), size=n, replace=False)
            feats = feats[sel]
        X_parts.append(feats)
        y_parts.append(np.full(n, idx, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = float(clf.score(X_test, y_test))
    cm = np.zeros((len(domains), len(domains)), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    return {
        "domains": domains,
        "accuracy": accuracy,
        "chance": 1.0 / len(domains),
        "confusion_matrix": cm.tolist(),
    }
