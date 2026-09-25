"""
Validation-calibrated rejection threshold: tau = 95th percentile of
unknownness score u(x) over the CIFAR-10 VALIDATION set (known examples
only). Accept x when u(x) <= tau -- by construction this accepts ~95% of
the validation known examples (a 95%-TPR operating point, using only
known validation data to pick tau, per spec). Applying tau to CIFAR-10
test / CIFAR-100 near / CIFAR-100 far then reports the ACHIEVED
acceptance/rejection rates at this one fixed operating point.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(u_val: np.ndarray, target_tpr: float = 0.95) -> float:
    return float(np.percentile(u_val, target_tpr * 100))


def acceptance_rate(u: np.ndarray, tau: float) -> float:
    """Fraction of examples with u(x) <= tau (i.e. accepted as known)."""
    return float((u <= tau).mean())


def fpr_at_threshold(u_unknown: np.ndarray, tau: float) -> float:
    """
    FPR@95TPR: fraction of UNKNOWN examples incorrectly accepted
    (u(x) <= tau) under the val-calibrated threshold -- equivalently
    1 - rejection_rate.
    """
    return float((u_unknown <= tau).mean())
