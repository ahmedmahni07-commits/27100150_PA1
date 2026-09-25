"""u_Energy(x) = -log sum_k exp(z_k(x)): negative log-sum-exp ("free energy") score."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def compute_energy(logits: np.ndarray) -> np.ndarray:
    """logits: (N, K). Returns u_Energy: (N,), larger = more novel."""
    return -logsumexp(logits, axis=1)
