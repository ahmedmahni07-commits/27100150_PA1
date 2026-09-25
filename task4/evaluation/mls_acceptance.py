"""CIFAR-10 test acceptance and near/far rejection at the 95%-val MLS threshold
for every trained model (numpy only, reads task4/cache/*.npz).
Usage: python -m task4.evaluation.mls_acceptance"""
import json
import numpy as np

out = {}
for m in ["vanilla", "gcsc", "proser"]:
    d = np.load(f"task4/cache/{m}_outputs.npz", allow_pickle=True)
    u = lambda s: -d[s + "__known_logits"].max(1)
    tau = float(np.percentile(u("cifar10_val"), 95))
    out[m] = {"tau": tau, "test_acceptance": float((u("cifar10_test") <= tau).mean()),
              "near_rejection": float((u("cifar100_near") > tau).mean()),
              "far_rejection": float((u("cifar100_far") > tau).mean())}
json.dump(out, open("task4/results/mls_acceptance.json", "w"), indent=2)
print(json.dumps(out, indent=1))
