"""u_MSP(x) = 1 - max_k p_k(x): normalized softmax-confidence unknownness score."""

from __future__ import annotations

import numpy as np


def compute_msp(logits: np.ndarray) -> np.ndarray:
    """logits: (N, K). Returns u_MSP: (N,), larger = more novel."""
    shifted = logits - logits.max(axis=1, keepdims=True)  # numerically stable softmax
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return 1.0 - probs.max(axis=1)
