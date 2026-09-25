"""
Step 5: per-class target accuracy, and which classes improve/degrade most
relative to Source-only. Target labels are used ONLY here, at final
analysis time -- never during training or checkpoint selection.
"""

from __future__ import annotations

import numpy as np


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Length-num_classes array; NaN for a class absent from y_true."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            accs[c] = (y_pred[mask] == c).mean()
    return accs


def compare_to_baseline(
    per_class_acc_method: np.ndarray, per_class_acc_source_only: np.ndarray, top_k: int = 3
) -> dict:
    """
    Per-class deltas (method - source_only) plus the top-k most improved
    and most degraded classes by index, for "identify the classes with the
    largest improvement and degradation relative to Source-only".
    """
    deltas = per_class_acc_method - per_class_acc_source_only
    valid = ~np.isnan(deltas)

    # Most improved: largest positive delta first. NaNs sink to the bottom
    # before reversing by treating them as -inf.
    improve_order = np.argsort(np.where(valid, deltas, -np.inf))[::-1]
    most_improved = improve_order[:top_k].tolist()

    # Most degraded: most negative delta first. NaNs pushed to the end by
    # treating them as +inf.
    degrade_order = np.argsort(np.where(valid, deltas, np.inf))
    most_degraded = degrade_order[:top_k].tolist()

    return {
        "per_class_delta": deltas.tolist(),
        "most_improved_classes": most_improved,
        "most_degraded_classes": most_degraded,
    }


def confusion_matrix_for_class(
    y_true: np.ndarray, y_pred: np.ndarray, class_idx: int, num_classes: int = 7
) -> np.ndarray:
    """
    For inspecting a flagged class's dominant confusions: returns a
    length-num_classes row -- what predicted label true-class `class_idx`
    examples actually received.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true == class_idx
    row = np.zeros(num_classes, dtype=int)
    for pred in y_pred[mask]:
        row[pred] += 1
    return row
