"""
Entry point: python -m task4.evaluate_osr

Run ONLY after:
    python -m task4.train --config task4/configs/vanilla.yaml
    python -m task4.train --config task4/configs/gcsc.yaml
    python -m task4.train --config task4/configs/proser.yaml
    python -m task4.extract_outputs --method vanilla
    python -m task4.extract_outputs --method gcsc
    python -m task4.extract_outputs --method proser

Produces Task 4's Required Evidence:
  1. MSP/MLS/Energy/Mahalanobis on the frozen VANILLA model: near/far/
     all-unknown AUROC + validation-calibrated (95th-percentile) rejection.
  2. Vanilla/GCSC/PROSER comparison: CSA + near/far OSR metrics, MLS as
     the common score, plus a PROSER-placeholder-score row.
  3. A compact score-distribution figure for MSP, MLS, and Mahalanobis.
  4. 3 near-unknown + 3 far-unknown incorrectly-accepted failure examples
     (vanilla MLS threshold).

CIFAR-100 unknown examples are read here for the FIRST time in this
pipeline (via extract_outputs.py's cache) -- they never influenced
training, checkpoint selection, score design, or threshold selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task4.evaluation.failure_analysis import find_incorrectly_accepted
from task4.evaluation.metrics import (
    auroc_known_vs_unknown,
    closed_set_accuracy,
    compute_proser_placeholder_score,
)
from task4.evaluation.thresholds import acceptance_rate, calibrate_threshold, fpr_at_threshold
from task4.scores.energy import compute_energy
from task4.scores.mahalanobis import compute_mahalanobis, fit_class_gaussians
from task4.scores.mls import compute_mls
from task4.scores.msp import compute_msp

CACHE_DIR = Path("task4/cache")
RESULTS_DIR = Path("task4/results")


def _load_cache(run_name: str) -> dict:
    path = CACHE_DIR / f"{run_name}_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached outputs at {path} -- run "
            f"`python -m task4.extract_outputs --method {run_name}` first."
        )
    raw = np.load(path, allow_pickle=True)
    out: dict = {}
    for key in raw.files:
        split_name, field = key.split("__", 1)
        out.setdefault(split_name, {})[field] = raw[key]
    return out


def _all_scores(logits: np.ndarray, features: np.ndarray, class_means, shared_precision) -> dict:
    return {
        "MSP": compute_msp(logits),
        "MLS": compute_mls(logits),
        "Energy": compute_energy(logits),
        "Mahalanobis": compute_mahalanobis(features, class_means, shared_precision),
    }


def build_vanilla_score_table(vanilla_cache: dict) -> dict:
    class_means, shared_precision = fit_class_gaussians(
        vanilla_cache["cifar10_train"]["features"], vanilla_cache["cifar10_train"]["labels"]
    )

    test = vanilla_cache["cifar10_test"]
    val = vanilla_cache["cifar10_val"]
    near = vanilla_cache["cifar100_near"]
    far = vanilla_cache["cifar100_far"]

    scores_test = _all_scores(test["known_logits"], test["features"], class_means, shared_precision)
    scores_val = _all_scores(val["known_logits"], val["features"], class_means, shared_precision)
    scores_near = _all_scores(near["known_logits"], near["features"], class_means, shared_precision)
    scores_far = _all_scores(far["known_logits"], far["features"], class_means, shared_precision)

    table = {}
    for score_name in ["MSP", "MLS", "Energy", "Mahalanobis"]:
        u_val, u_test = scores_val[score_name], scores_test[score_name]
        u_near, u_far = scores_near[score_name], scores_far[score_name]
        u_all_unknown = np.concatenate([u_near, u_far])

        tau = calibrate_threshold(u_val, target_tpr=0.95)

        table[score_name] = {
            "auroc_near": auroc_known_vs_unknown(u_test, u_near),
            "auroc_far": auroc_known_vs_unknown(u_test, u_far),
            "auroc_all": auroc_known_vs_unknown(u_test, u_all_unknown),
            "threshold_tau": tau,
            "test_acceptance_rate": acceptance_rate(u_test, tau),
            "near_fpr_at_95tpr": fpr_at_threshold(u_near, tau),
            "far_fpr_at_95tpr": fpr_at_threshold(u_far, tau),
            "all_fpr_at_95tpr": fpr_at_threshold(u_all_unknown, tau),
        }

    return {
        "table": table,
        "class_means": class_means,
        "shared_precision": shared_precision,
        "scores_val": scores_val,
        "scores_test": scores_test,
        "scores_near": scores_near,
        "scores_far": scores_far,
    }


def build_method_comparison_table(caches: dict) -> dict:
    """Vanilla/GCSC/PROSER using CSA + near/far/all OSR metrics, MLS as
    the common score; PROSER also gets its own placeholder-score row."""
    table = {}
    for method in ["vanilla", "gcsc", "proser"]:
        cache = caches[method]
        test, near, far, val = cache["cifar10_test"], cache["cifar100_near"], cache["cifar100_far"], cache["cifar10_val"]

        # CSA computed from the ten known-class logits only, per spec
        # ("compute CSA using only the ten known-class logits").
        csa = closed_set_accuracy(test["known_logits"], test["labels"])

        u_val_mls = compute_mls(val["known_logits"])
        u_test_mls = compute_mls(test["known_logits"])
        u_near_mls = compute_mls(near["known_logits"])
        u_far_mls = compute_mls(far["known_logits"])
        u_all_mls = np.concatenate([u_near_mls, u_far_mls])
        tau_mls = calibrate_threshold(u_val_mls, target_tpr=0.95)

        table[method] = {
            "score": "MLS",
            "csa": csa,
            "auroc_near": auroc_known_vs_unknown(u_test_mls, u_near_mls),
            "auroc_far": auroc_known_vs_unknown(u_test_mls, u_far_mls),
            "auroc_all": auroc_known_vs_unknown(u_test_mls, u_all_mls),
            "near_fpr_at_95tpr": fpr_at_threshold(u_near_mls, tau_mls),
            "far_fpr_at_95tpr": fpr_at_threshold(u_far_mls, tau_mls),
            "all_fpr_at_95tpr": fpr_at_threshold(u_all_mls, tau_mls),
        }

        if method == "proser":
            u_val_ph = compute_proser_placeholder_score(val["known_logits"], val["dummy_logits"])
            u_test_ph = compute_proser_placeholder_score(test["known_logits"], test["dummy_logits"])
            u_near_ph = compute_proser_placeholder_score(near["known_logits"], near["dummy_logits"])
            u_far_ph = compute_proser_placeholder_score(far["known_logits"], far["dummy_logits"])
            u_all_ph = np.concatenate([u_near_ph, u_far_ph])
            tau_ph = calibrate_threshold(u_val_ph, target_tpr=0.95)

            table["proser_placeholder"] = {
                "score": "PROSER-placeholder",
                "csa": csa,  # same known-class logits, so identical CSA to the MLS row
                "auroc_near": auroc_known_vs_unknown(u_test_ph, u_near_ph),
                "auroc_far": auroc_known_vs_unknown(u_test_ph, u_far_ph),
                "auroc_all": auroc_known_vs_unknown(u_test_ph, u_all_ph),
                "near_fpr_at_95tpr": fpr_at_threshold(u_near_ph, tau_ph),
                "far_fpr_at_95tpr": fpr_at_threshold(u_far_ph, tau_ph),
                "all_fpr_at_95tpr": fpr_at_threshold(u_all_ph, tau_ph),
            }

    return table


def plot_score_distributions(vanilla_eval: dict, out_path: Path) -> None:
    """One compact multi-panel figure: MSP, MLS, Mahalanobis score
    distributions (known test vs. near-unknown vs. far-unknown)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, score_name in zip(axes, ["MSP", "MLS", "Mahalanobis"]):
        u_test = vanilla_eval["scores_test"][score_name]
        u_near = vanilla_eval["scores_near"][score_name]
        u_far = vanilla_eval["scores_far"][score_name]
        bins = np.histogram_bin_edges(np.concatenate([u_test, u_near, u_far]), bins=40)
        ax.hist(u_test, bins=bins, alpha=0.5, label="known (CIFAR-10 test)", density=True)
        ax.hist(u_near, bins=bins, alpha=0.5, label="near-unknown", density=True)
        ax.hist(u_far, bins=bins, alpha=0.5, label="far-unknown", density=True)
        ax.set_title(score_name)
        ax.set_xlabel("unknownness score u(x)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_failure_analysis(vanilla_cache: dict, vanilla_eval: dict) -> dict:
    tau = vanilla_eval["table"]["MLS"]["threshold_tau"]
    near, far = vanilla_cache["cifar100_near"], vanilla_cache["cifar100_far"]

    near_failures = find_incorrectly_accepted(
        vanilla_eval["scores_near"]["MLS"], near["known_logits"], near.get("class_names"), tau, top_k=3
    )
    far_failures = find_incorrectly_accepted(
        vanilla_eval["scores_far"]["MLS"], far["known_logits"], far.get("class_names"), tau, top_k=3
    )
    return {"near_unknown_failures": near_failures, "far_unknown_failures": far_failures}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached outputs...")
    vanilla_cache = _load_cache("vanilla")
    gcsc_cache = _load_cache("gcsc")
    proser_cache = _load_cache("proser")

    print("Table 1: MSP/MLS/Energy/Mahalanobis on frozen Vanilla model...")
    vanilla_eval = build_vanilla_score_table(vanilla_cache)

    print("Table 2: Vanilla/GCSC/PROSER comparison (MLS + PROSER placeholder)...")
    comparison_table = build_method_comparison_table(
        {"vanilla": vanilla_cache, "gcsc": gcsc_cache, "proser": proser_cache}
    )

    print("Score-distribution figure...")
    fig_path = RESULTS_DIR / "score_distributions.png"
    plot_score_distributions(vanilla_eval, fig_path)

    print("Failure analysis (vanilla MLS threshold)...")
    failures = run_failure_analysis(vanilla_cache, vanilla_eval)

    results = {
        "vanilla_score_comparison": vanilla_eval["table"],
        "method_comparison_mls": comparison_table,
        "failure_analysis": failures,
    }
    out_path = RESULTS_DIR / "evaluate_osr_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print("Table 1 -- Vanilla model, all four scores:")
    print(
        f"{'Score':<14}{'AUROC-near':>12}{'AUROC-far':>11}{'AUROC-all':>11}"
        f"{'TestAccept':>12}{'NearFPR@95':>12}{'FarFPR@95':>11}"
    )
    for score_name, r in vanilla_eval["table"].items():
        print(
            f"{score_name:<14}{r['auroc_near']:>12.3f}{r['auroc_far']:>11.3f}{r['auroc_all']:>11.3f}"
            f"{r['test_acceptance_rate']*100:>11.1f}%{r['near_fpr_at_95tpr']*100:>11.1f}%"
            f"{r['far_fpr_at_95tpr']*100:>10.1f}%"
        )
    print("-" * 100)
    print("Table 2 -- Vanilla/GCSC/PROSER comparison:")
    print(f"{'Method':<20}{'Score':<20}{'CSA':>8}{'AUROC-near':>12}{'AUROC-far':>11}{'AUROC-all':>11}")
    for method, r in comparison_table.items():
        print(
            f"{method:<20}{r['score']:<20}{r['csa']*100:>7.1f}%"
            f"{r['auroc_near']:>12.3f}{r['auroc_far']:>11.3f}{r['auroc_all']:>11.3f}"
        )
    print("=" * 100)
    print(f"\nFull results written to {out_path}")
    print(f"Score-distribution figure written to {fig_path}")


if __name__ == "__main__":
    main()
