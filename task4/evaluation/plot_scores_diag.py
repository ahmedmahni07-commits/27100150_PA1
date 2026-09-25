"""MSP / MLS / diagonal-Mahalanobis score histograms on the frozen Vanilla model
(same protocol as evaluate_osr; diagonal shared covariance as in the handout).
Usage: python -m task4.evaluation.plot_scores_diag"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("task4/cache/vanilla_outputs.npz", allow_pickle=True)
f, y = d["cifar10_train__features"].astype(np.float64), d["cifar10_train__labels"]
mu = np.stack([f[y == c].mean(0) for c in range(10)])
prec = 1.0 / (np.concatenate([f[y == c] - mu[c] for c in range(10)]) ** 2).mean(0) + 0.0
prec = 1.0 / (1.0 / prec + 1e-6)

def scores(s):
    z = d[s + "__known_logits"].astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    feat = d[s + "__features"].astype(np.float64)
    maha = np.min([(((feat - m) ** 2) * prec).sum(1) for m in mu], 0)
    return {"MSP": 1 - p.max(1), "MLS": -z.max(1), "Mahalanobis (diagonal)": maha}

S = {lab: scores(k) for k, lab in [("cifar10_test", "known (CIFAR-10 test)"),
                                   ("cifar100_near", "near unknown"), ("cifar100_far", "far unknown")]}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, name in zip(ax, ["MSP", "MLS", "Mahalanobis (diagonal)"]):
    allv = np.concatenate([S[g][name] for g in S])
    bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 40)
    for g in S:
        a.hist(S[g][name], bins=bins, density=True, alpha=0.5, label=g)
    a.set_title(name); a.set_xlabel("unknownness score u(x)"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("task4/results/score_distributions_diag.png", dpi=200)
print("saved")
