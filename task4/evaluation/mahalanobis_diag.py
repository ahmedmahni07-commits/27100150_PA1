"""
Spec-compliant Mahalanobis row: ONE shared DIAGONAL covariance (+1e-6 on the
diagonal) from unaugmented CIFAR-10 train features. scores/mahalanobis.py uses
the full shared covariance; this script reports the diagonal variant on the
same cached vanilla features, same AUROC/threshold protocol, numpy only.
Usage: python -m task4.evaluation.mahalanobis_diag
"""
import json
import numpy as np


def auroc(neg, pos):  # P(u_unknown > u_known), ties counted half
    x = np.concatenate([neg, pos]); r = np.empty(len(x))
    order = np.argsort(x, kind="mergesort"); r[order] = np.arange(1, len(x) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    out = {}
    for method in ["vanilla", "gcsc", "proser"]:
        d = np.load(f"task4/cache/{method}_outputs.npz", allow_pickle=True)
        f, y = d["cifar10_train__features"].astype(np.float64), d["cifar10_train__labels"]
        mu = np.stack([f[y == c].mean(0) for c in range(10)])
        cen = np.concatenate([f[y == c] - mu[c] for c in range(10)])
        prec = 1.0 / ((cen ** 2).mean(0) + 1e-6)
        u = lambda s: np.min([(((d[f"{s}__features"].astype(np.float64) - m) ** 2) * prec).sum(1) for m in mu], 0)
        val, te, ne, fa = (u(s) for s in ["cifar10_val", "cifar10_test", "cifar100_near", "cifar100_far"])
        tau = np.percentile(val, 95)
        out[method] = {"auroc_near": auroc(te, ne), "auroc_far": auroc(te, fa), "auroc_all": auroc(te, np.concatenate([ne, fa])),
                       "threshold_tau": float(tau), "test_acceptance_rate": float((te <= tau).mean()),
                       "near_fpr_at_95tpr": float((ne <= tau).mean()), "far_fpr_at_95tpr": float((fa <= tau).mean())}
        print(method, {k: round(v, 4) for k, v in out[method].items()})
    json.dump(out, open("task4/results/mahalanobis_diag_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
