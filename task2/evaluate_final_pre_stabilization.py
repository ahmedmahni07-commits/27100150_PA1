"""
Entry point: python -m task2.evaluate_final_pre_stabilization

Re-runs the Common Evaluation and Alignment Diagnostic (Step 5) against
the PRE-STABILIZATION DANN/CDAN checkpoints -- the as-specified,
unclipped/un-normalized run, saved in
task2/results/pre_stabilization/checkpoints/ before the gradient-clipping
+ discriminator-normalization deviation documented in task2/DEVIATIONS.md
was applied. This is the run that should be reported as the MAIN Common
Evaluation results, since the deviation is a separately-documented
exploration, not a replacement for the as-specified methodology.

Source-only and DAN are untouched by that deviation (their configs never
set grad_clip_norm, and the normalization only affects DANN's/CDAN's
domain-discriminator branch, which evaluation never touches anyway --
evaluate_one_method only runs backbone+head forward passes). So their
CURRENT checkpoints in task2/results/checkpoints/ are the same weights
either way and are reused as-is; only dann/cdan point at
pre_stabilization/checkpoints/ instead.

Writes task2/results/pre_stabilization/evaluate_final_results.json --
deliberately a SEPARATE file from the current (stabilized)
task2/results/evaluate_final_results.json, so neither run clobbers the
other and both are directly comparable side by side.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from common.pacs import get_backbone_transforms
from common.pacs_protocol import build_or_load_source_splits, build_or_load_target_pool
from task2.evaluate_final import METHOD_CONFIGS, evaluate_one_method, get_device
from task2.evaluation.class_analysis import compare_to_baseline, confusion_matrix_for_class
from task2.train import load_config

PRE_STABILIZATION_CKPT_DIR = "task2/results/pre_stabilization/checkpoints"
# Only dann/cdan checkpoints predate the deviation; source_only/dan are unaffected.
METHODS_NEEDING_PRE_STABILIZATION_CKPT = {"dann", "cdan"}


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    base_cfg = load_config("task2/configs/base.yaml")
    source_splits = build_or_load_source_splits(base_cfg["data"]["pacs_root"])
    target_samples = build_or_load_target_pool(base_cfg["data"]["pacs_root"])
    eval_transform = get_backbone_transforms("val", base_cfg["data"]["resize"], base_cfg["data"]["crop"])

    results = {}
    for method, config_path in METHOD_CONFIGS.items():
        cfg = load_config(config_path)
        if method in METHODS_NEEDING_PRE_STABILIZATION_CKPT:
            cfg = copy.deepcopy(cfg)
            cfg["logging"]["checkpoint_dir"] = PRE_STABILIZATION_CKPT_DIR
            print(f"\n=== Evaluating {method} (PRE-stabilization checkpoint, from {PRE_STABILIZATION_CKPT_DIR}) ===")
        else:
            print(f"\n=== Evaluating {method} (unaffected by the deviation -- current checkpoint) ===")
        results[method] = evaluate_one_method(
            method, cfg, device, source_splits, target_samples, eval_transform
        )

    baseline_acc = results["source_only"]["target_accuracy"]
    baseline_per_class = np.array(results["source_only"]["per_class_target_accuracy"])
    for method in results:
        results[method]["target_accuracy_change_vs_source_only"] = (
            results[method]["target_accuracy"] - baseline_acc
        )
        if method != "source_only":
            method_per_class = np.array(results[method]["per_class_target_accuracy"])
            comparison = compare_to_baseline(method_per_class, baseline_per_class)
            results[method]["per_class_comparison_vs_source_only"] = comparison

            # Same "inspect their dominant confusions" evidence as evaluate_final.py,
            # but for the as-specified (unclipped) checkpoints.
            num_classes = len(baseline_per_class)
            flagged_classes = sorted(set(
                comparison["most_improved_classes"] + comparison["most_degraded_classes"]
            ))
            source_only_y_true = np.array(results["source_only"]["target_y_true"])
            source_only_y_pred = np.array(results["source_only"]["target_y_pred"])
            method_y_true = np.array(results[method]["target_y_true"])
            method_y_pred = np.array(results[method]["target_y_pred"])
            results[method]["flagged_class_confusions"] = {
                str(c): {
                    "source_only_predicted_as": confusion_matrix_for_class(
                        source_only_y_true, source_only_y_pred, c, num_classes
                    ).tolist(),
                    f"{method}_predicted_as": confusion_matrix_for_class(
                        method_y_true, method_y_pred, c, num_classes
                    ).tolist(),
                }
                for c in flagged_classes
            }

    out_dir = Path("task2/results/pre_stabilization")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluate_final_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 92)
    print(f"{'Method':<14}{'Src Acc':>10}{'Src F1':>9}{'Tgt Acc':>10}{'Tgt F1':>9}{'D Tgt Acc':>12}{'DomSep':>10}")
    print("-" * 92)
    for method in ["source_only", "dan", "dann", "cdan"]:
        r = results[method]
        print(
            f"{method:<14}"
            f"{r['mean_source_accuracy']*100:>9.1f}%"
            f"{r['mean_source_macro_f1']:>9.3f}"
            f"{r['target_accuracy']*100:>9.1f}%"
            f"{r['target_macro_f1']:>9.3f}"
            f"{r['target_accuracy_change_vs_source_only']*100:>+11.1f}%"
            f"{r['domain_separability']*100:>9.1f}%"
        )
    print("=" * 92)
    print(f"\nFull results (including confusion matrices) written to {out_path}")


if __name__ == "__main__":
    main()
