"""
Step 6 failure analysis: using a given threshold (the vanilla MLS
threshold, per spec), find incorrectly-accepted near/far unknown examples
(u(x) <= tau despite being unknown), returning enough info to report
"unknown class, predicted CIFAR-10 class, score, threshold" per the
Required Evidence.
"""

from __future__ import annotations

import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def find_incorrectly_accepted(
    u_unknown: np.ndarray,
    known_logits_unknown: np.ndarray,
    class_names_unknown,
    tau: float,
    top_k: int = 3,
) -> list[dict]:
    """
    Returns up to `top_k` incorrectly-accepted examples (u(x) <= tau),
    sorted by CONFIDENCE (lowest u(x) first -- i.e. the most confidently
    "known-looking" unknowns, the most informative failures to inspect),
    each with its unknown fine-class name, predicted CIFAR-10 class, the
    score, and the threshold.
    """
    accepted_mask = u_unknown <= tau
    accepted_idx = np.nonzero(accepted_mask)[0]
    if len(accepted_idx) == 0:
        return []

    order = accepted_idx[np.argsort(u_unknown[accepted_idx])]  # most confident first
    selected = order[:top_k]

    preds = known_logits_unknown.argmax(axis=1)
    results = []
    for i in selected:
        results.append({
            "unknown_class": str(class_names_unknown[i]) if class_names_unknown is not None else None,
            "predicted_cifar10_class": CIFAR10_CLASSES[int(preds[i])],
            "score": float(u_unknown[i]),
            "threshold": float(tau),
        })
    return results
