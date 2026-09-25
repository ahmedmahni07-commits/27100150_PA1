"""
Shared evaluation metrics for Task 4: closed-set accuracy (CSA), AUROC
for known-vs-unknown, and the PROSER placeholder-based unknownness score
(computed here in numpy from cached logits, mirroring
task4.methods.proser.placeholder_score's torch version exactly, since
evaluate_osr.py works entirely from extract_outputs.py's cached arrays,
not a live model).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def closed_set_accuracy(known_logits: np.ndarray, labels: np.ndarray) -> float:
    preds = known_logits.argmax(axis=1)
    return float((preds == labels).mean())


def auroc_known_vs_unknown(u_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """
    u_known/u_unknown: unknownness scores (larger = more novel) for known
    (CIFAR-10 test) and unknown (CIFAR-100 near/far/all) examples
    respectively. y=0 for known, y=1 for unknown -- AUROC computed
    directly on u (already oriented so larger => more likely the positive
    "unknown" class), no negation needed.
    """
    y_true = np.concatenate([np.zeros(len(u_known)), np.ones(len(u_unknown))])
    scores = np.concatenate([u_known, u_unknown])
    return float(roc_auc_score(y_true, scores))


def compute_proser_placeholder_score(known_logits: np.ndarray, dummy_logits: np.ndarray) -> np.ndarray:
    """
    PROSER's own placeholder-based unknownness score: softmax probability
    mass on the collapsed placeholder position (max over dummy logits),
    over the (K+1)-way vector [known_logits, max(dummy_logits)]. Larger =
    more novel, same u(x) convention as MSP/MLS/Energy/Mahalanobis.
    """
    placeholder_logit = dummy_logits.max(axis=1, keepdims=True)
    full_logits = np.concatenate([known_logits, placeholder_logit], axis=1)
    shifted = full_logits - full_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, -1]
