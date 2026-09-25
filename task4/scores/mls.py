"""u_MLS(x) = -max_k z_k(x): negative max-logit score (retains absolute logit magnitude, unlike MSP's normalized softmax)."""

from __future__ import annotations

import numpy as np


def compute_mls(logits: np.ndarray) -> np.ndarray:
    """logits: (N, K). Returns u_MLS: (N,), larger = more novel."""
    return -logits.max(axis=1)
