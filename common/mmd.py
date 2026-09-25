"""
Shared Maximum Mean Discrepancy (MMD) implementation.

Used by Task 2's DAN and (later) Task 3's DAN-DG — kept here so both tasks
import the identical kernel construction, per the spec: "Use the same MMD
implementation and kernel construction as Task 2."

Design choices (documented here since the spec leaves them to you):
  - Kernel: sum of three RBF kernels, bandwidths = {0.5, 1, 2} x the median
    pairwise squared distance of the current combined (x, y) batch.
  - The bandwidth statistic (the median) is DETACHED from the autograd graph
    before use — this is standard practice in DAN/JAN-style implementations
    (see e.g. jindongwang/transferlearning's guassian_kernel): gradients flow
    through the kernel's *argument* (the features), not through the bandwidth
    normalizer itself, which keeps optimization stable.
  - MMD^2 estimator: the biased/V-statistic estimator
    mean(Kxx) + mean(Kyy) - 2*mean(Kxy). This includes each set's diagonal
    (self-similarity = 1) in the mean, which is simpler and standard; note
    this choice in your README if asked to justify it.
"""

import torch


def _pairwise_sq_dists(feats: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distances over `feats` (N, D), via
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b (matmul-based) instead of
    torch.cdist.

    MPS (Apple GPU) has no backward implementation for `aten::_cdist_backward`
    (torch.cdist's own compute_mode="use_mm_for_euclid_dist_if_necessary"
    doesn't help -- autograd still records a cdist node, not the underlying
    matmul), so anything using cdist upstream of a loss.backward() call
    raises `NotImplementedError` on an MPS device. This formula is
    numerically equivalent and built entirely out of ops MPS does support
    (mul, sum, matmul), so it works identically on CPU/CUDA/MPS.

    Clamped at 0 to guard against tiny negative values from floating-point
    cancellation (e.g. the exact-zero diagonal).
    """
    sq_norms = (feats * feats).sum(dim=1)
    sq_dists = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (feats @ feats.t())
    return sq_dists.clamp(min=0.0)


def median_pairwise_sq_distance(feats: torch.Tensor) -> torch.Tensor:
    """
    Median of pairwise squared Euclidean distances over `feats` (N, D),
    computed on the off-diagonal (i != j) pairs only, and detached from
    the autograd graph (used only as a bandwidth statistic).
    """
    n = feats.size(0)
    if n < 2:
        return torch.tensor(1.0, device=feats.device, dtype=feats.dtype)

    sq_dists = _pairwise_sq_dists(feats)
    iu = torch.triu_indices(n, n, offset=1, device=feats.device)
    off_diag = sq_dists[iu[0], iu[1]]

    median = torch.median(off_diag)
    median = torch.clamp(median, min=1e-8)  # guard against a degenerate all-identical batch
    return median.detach()


def rbf_kernel_sum(
    x: torch.Tensor, y: torch.Tensor, bandwidth_multipliers=(0.5, 1.0, 2.0)
) -> torch.Tensor:
    """
    Sum of RBF kernels exp(-sq_dist / bandwidth) for each bandwidth in
    {multiplier * median_pairwise_sq_distance(combined) for multiplier in
    bandwidth_multipliers}, evaluated over every pair of rows of
    cat([x, y], dim=0).

    Returns the full (Nx+Ny, Nx+Ny) kernel matrix (already summed over the
    three bandwidths) so mmd2() can slice out the XX/YY/XY blocks.
    """
    combined = torch.cat([x, y], dim=0)
    sq_dists = _pairwise_sq_dists(combined)
    median = median_pairwise_sq_distance(combined)

    kernel_matrix = torch.zeros_like(sq_dists)
    for multiplier in bandwidth_multipliers:
        bandwidth = multiplier * median
        kernel_matrix = kernel_matrix + torch.exp(-sq_dists / bandwidth)
    return kernel_matrix


def mmd2(
    x: torch.Tensor, y: torch.Tensor, bandwidth_multipliers=(0.5, 1.0, 2.0)
) -> torch.Tensor:
    """
    Squared MMD between two feature sets x (Nx, D) and y (Ny, D), using the
    RBF-sum kernel trick (biased estimator, see module docstring).

    L_DAN = L_cls + lambda_MMD * mmd2(source_feats, target_feats)

    For Task 3's DAN-DG, call this once per unordered source-domain pair
    and average, per that task's spec.
    """
    nx, ny = x.size(0), y.size(0)
    K = rbf_kernel_sum(x, y, bandwidth_multipliers)
    Kxx = K[:nx, :nx]
    Kyy = K[nx:, nx:]
    Kxy = K[:nx, nx:]
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()