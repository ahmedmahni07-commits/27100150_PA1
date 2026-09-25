"""
Post-hoc failure analysis only (run after every checkpoint, score and
threshold was frozen): for each CIFAR-100 unknown class, the fraction accepted
by the validation-calibrated MLS threshold (95th pct of u on CIFAR-10 val) and
the CIFAR-10 labels that absorb the accepted examples. Also Spearman rank
agreement between scores on the pooled unknown+known test set.
Usage: python -m task4.evaluation.per_class_acceptance
"""
import json
from collections import Counter
import numpy as np

C10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def lse(z):
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def scores(d, split, mu, prec):
    z = d[f"{split}__known_logits"].astype(np.float64)
    f = d[f"{split}__features"].astype(np.float64)
    p = np.exp(z - lse(z)[:, None])
    maha = np.min([(((f - m) ** 2) * prec).sum(1) for m in mu], axis=0)
    return {"MSP": 1 - p.max(1), "MLS": -z.max(1), "Energy": -lse(z), "Mahalanobis": maha}


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def main():
    out = {}
    for method in ["vanilla", "gcsc", "proser"]:
        d = np.load(f"task4/cache/{method}_outputs.npz", allow_pickle=True)
        ftr, ytr = d["cifar10_train__features"].astype(np.float64), d["cifar10_train__labels"]
        mu = [ftr[ytr == c].mean(0) for c in range(10)]
        var = np.mean([((ftr[ytr == c] - mu[c]) ** 2).mean(0) for c in range(10)], axis=0) + 1e-6
        # pooled (shared) diagonal covariance, class-size weighted (classes are balanced)
        prec = 1.0 / var
        S = {s: scores(d, s, mu, prec) for s in ["cifar10_val", "cifar10_test", "cifar100_near", "cifar100_far"]}
        res = {}
        for sc in ["MSP", "MLS", "Energy", "Mahalanobis"]:
            tau = np.percentile(S["cifar10_val"][sc], 95)
            r = {}
            for grp in ["cifar100_near", "cifar100_far"]:
                names = d[f"{grp}__class_names"]
                pred = d[f"{grp}__known_logits"].argmax(1)
                acc = S[grp][sc] <= tau
                per = {}
                for cn in sorted(set(names)):
                    mk = names == cn
                    per[str(cn)] = {"accept_rate": float(acc[mk].mean()),
                                    "absorbed_by": {C10[k]: int(v) for k, v in Counter(pred[mk & acc]).most_common(3)}}
                r[grp] = per
            res[sc] = r
        allu = {sc: np.concatenate([S[g][sc] for g in ["cifar10_test", "cifar100_near", "cifar100_far"]]) for sc in S["cifar10_test"]}
        sp = {}
        ks = list(allu)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                sp[f"{ks[i]}~{ks[j]}"] = float(np.corrcoef(rank(allu[ks[i]]), rank(allu[ks[j]]))[0, 1])
        res["spearman_all_test"] = sp
        # top-1 agreement: near unknowns flagged by Mahalanobis but not MLS and vice versa
        taus = {sc: np.percentile(S["cifar10_val"][sc], 95) for sc in ["MLS", "Mahalanobis"]}
        for grp in ["cifar100_near", "cifar100_far"]:
            rm = S[grp]["MLS"] > taus["MLS"]; rh = S[grp]["Mahalanobis"] > taus["Mahalanobis"]
            res[f"reject_overlap_{grp}"] = {"both": int((rm & rh).sum()), "MLS_only": int((rm & ~rh).sum()),
                                           "Maha_only": int((~rm & rh).sum()), "neither": int((~rm & ~rh).sum())}
        out[method] = res
    json.dump(out, open("task4/results/per_class_acceptance.json", "w"), indent=2)
    v = out["vanilla"]
    for grp in ["cifar100_near", "cifar100_far"]:
        print(grp)
        for cn, x in sorted(v["MLS"][grp].items(), key=lambda t: -t[1]["accept_rate"]):
            print(f"  {cn:14s} MLS-accept {x['accept_rate']:.2f}  MSP {v['MSP'][grp][cn]['accept_rate']:.2f}  Maha {v['Mahalanobis'][grp][cn]['accept_rate']:.2f}  absorbed {x['absorbed_by']}")
    for m in out:
        print(m, {k: round(v, 3) for k, v in out[m]["spearman_all_test"].items()})
        print(m, out[m]["reject_overlap_cifar100_near"], out[m]["reject_overlap_cifar100_far"])


if __name__ == "__main__":
    main()
