"""
Entry point: python -m task3.evaluate_sketch

Run ONLY after ERM/DAN-DG/SAM's checkpoints and configs are frozen. This is
the FIRST and ONLY place in Task 3 that Sketch is loaded at all (train.py
and selection/source_validation.py never import build_or_load_target_pool)
-- everything upstream of this script selected its checkpoint using
source-validation macro-F1 alone.

Produces (spec's Section 4, "Common Evaluation and Diagnostics"):
  - accuracy + macro-F1 per source validation domain, their mean and
    worst-domain values, and final Sketch accuracy/macro-F1, for
    ERM / DAN-DG / SAM
  - change in Sketch accuracy (and macro-F1) relative to ERM
  - source-domain separability (3-way Photo/Art/Cartoon, chance=33.3%)
  - local sharpness proxy (Delta_sharp, rho=0.05, fixed 96-example
    validation batch, seed 6304, model in EVAL mode) for all three models
  - per-class Sketch accuracy + most-improved/most-degraded vs. ERM
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.pacs_protocol import (
    build_or_load_source_splits,
    build_or_load_target_pool,
    make_source_eval_loader,
    make_target_eval_loader,
)
from common.pacs import get_backbone_transforms
from task2.evaluate_final import collect_features
from task3.evaluation import domain_metrics
from task3.evaluation.sharpness import build_fixed_sharpness_batch, compute_local_sharpness
from task3.evaluation.source_domain_separability import compute_source_domain_separability
from task2.evaluation.metrics import evaluate_loader
from task3.methods import dan_dg, erm, sam
from task3.train import get_device, load_config

METHOD_MODULES = {
    "erm": erm,
    "dan_dg": dan_dg,
    "sam": sam,
}

METHOD_CONFIGS = {
    "erm": "task3/configs/erm.yaml",
    "dan_dg": "task3/configs/dan_dg.yaml",
    "sam": "task3/configs/sam.yaml",
}

SHARPNESS_RHO = 0.05


def evaluate_one_method(
    method: str,
    cfg: dict,
    device: torch.device,
    source_splits: dict,
    target_samples: list,
    eval_transform,
    fixed_sharpness_batch: tuple,
    criterion: nn.CrossEntropyLoss,
    ckpt_name: str | None = None,
) -> dict:
    """
    `ckpt_name` overrides which checkpoint file to load (defaults to
    `method`) -- mirrors task2/evaluate_final.py's same parameter, kept
    for a future Task 3 controlled-study sweep (Section 5) to reuse this
    function without duplicating it.
    """
    method_module = METHOD_MODULES[method]
    model = method_module.build_model(cfg, device)

    ckpt_name = ckpt_name or method
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_path = checkpoint_dir / f"{ckpt_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} -- run "
            f"`python -m task3.train --config <its config>` first."
        )
    method_module.load_checkpoint(model, str(ckpt_path), device)

    # Source-validation performance, per domain -- and features, for the
    # source-domain separability diagnostic below.
    per_domain_val = {}
    features_by_domain = {}
    for domain, split in source_splits.items():
        loader = make_source_eval_loader(split, eval_transform)
        result = evaluate_loader(model, loader, device)
        per_domain_val[domain] = {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
        features_by_domain[domain] = collect_features(model, loader, device)

    accs = [v["accuracy"] for v in per_domain_val.values()]
    f1s = [v["macro_f1"] for v in per_domain_val.values()]
    mean_source_accuracy = float(np.mean(accs))
    mean_source_macro_f1 = float(np.mean(f1s))
    worst_source_accuracy = float(np.min(accs))
    worst_source_macro_f1 = float(np.min(f1s))

    # Sketch. This is the first time this method's checkpoint ever sees
    # Sketch -- it was never used to build/select this checkpoint.
    target_loader = make_target_eval_loader(target_samples, eval_transform)
    target_result = evaluate_loader(model, target_loader, device)

    # Source-domain separability: 3-way Photo/Art/Cartoon, chance=33.3%.
    domain_sep = compute_source_domain_separability(features_by_domain)

    # Local sharpness proxy: same fixed 96-example validation batch
    # (built once by the caller) and same rho for every method.
    delta_sharp = compute_local_sharpness(model, fixed_sharpness_batch, criterion, rho=SHARPNESS_RHO)

    per_class_acc = domain_metrics.per_class_accuracy(
        target_result["y_true"], target_result["y_pred"], num_classes=cfg["model"]["num_classes"]
    )

    return {
        "per_domain_val": per_domain_val,
        "mean_source_accuracy": mean_source_accuracy,
        "mean_source_macro_f1": mean_source_macro_f1,
        "worst_source_accuracy": worst_source_accuracy,
        "worst_source_macro_f1": worst_source_macro_f1,
        "sketch_accuracy": target_result["accuracy"],
        "sketch_macro_f1": target_result["macro_f1"],
        "source_domain_separability": domain_sep,
        "delta_sharp": delta_sharp,
        "per_class_sketch_accuracy": per_class_acc.tolist(),
    }


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    base_cfg = load_config("task3/configs/base.yaml")
    source_splits = build_or_load_source_splits(base_cfg["data"]["pacs_root"])
    # The ONLY call to build_or_load_target_pool anywhere in task3/.
    target_samples = build_or_load_target_pool(base_cfg["data"]["pacs_root"])
    eval_transform = get_backbone_transforms("val", base_cfg["data"]["resize"], base_cfg["data"]["crop"])
    criterion = nn.CrossEntropyLoss()

    # Built ONCE (seed 6304, 32/source domain) and reused unchanged for
    # every method, per spec -- the sharpness comparison is only
    # meaningful if all three methods are perturbed from the same point.
    fixed_sharpness_batch = build_fixed_sharpness_batch(
        source_splits, eval_transform, device, per_domain=32, seed=6304
    )

    results = {}
    for method, config_path in METHOD_CONFIGS.items():
        print(f"\n=== Evaluating {method} ===")
        cfg = load_config(config_path)
        results[method] = evaluate_one_method(
            method, cfg, device, source_splits, target_samples, eval_transform,
            fixed_sharpness_batch, criterion,
        )

    # Change in Sketch accuracy/macro-F1 relative to ERM.
    comparison = domain_metrics.build_comparison_table(
        {m: {"sketch": {"accuracy": r["sketch_accuracy"], "macro_f1": r["sketch_macro_f1"]}}
         for m, r in results.items()}
    )
    for method in results:
        results[method]["delta_sketch_accuracy_vs_erm"] = comparison[method].get("delta_accuracy_vs_erm")
        results[method]["delta_sketch_macro_f1_vs_erm"] = comparison[method].get("delta_macro_f1_vs_erm")

    # Per-class Sketch accuracy vs. ERM -- most improved/degraded classes,
    # available only now that configs are frozen and Sketch labels are in play.
    erm_per_class = np.array(results["erm"]["per_class_sketch_accuracy"])
    for method in results:
        if method == "erm":
            continue
        method_per_class = np.array(results[method]["per_class_sketch_accuracy"])
        results[method]["per_class_comparison_vs_erm"] = domain_metrics.compare_to_baseline(
            method_per_class, erm_per_class
        )

    results_dir = Path(base_cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluate_sketch_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 116)
    print(
        f"{'Method':<10}{'SrcAcc(mean)':>13}{'SrcAcc(worst)':>15}{'SrcF1(mean)':>13}"
        f"{'SketchAcc':>11}{'SketchF1':>10}{'D SketchAcc':>13}{'DomSep':>9}{'DeltaSharp':>12}"
    )
    print("-" * 116)
    for method in ["erm", "dan_dg", "sam"]:
        r = results[method]
        print(
            f"{method:<10}"
            f"{r['mean_source_accuracy']*100:>12.1f}%"
            f"{r['worst_source_accuracy']*100:>14.1f}%"
            f"{r['mean_source_macro_f1']:>13.3f}"
            f"{r['sketch_accuracy']*100:>10.1f}%"
            f"{r['sketch_macro_f1']:>10.3f}"
            f"{r['delta_sketch_accuracy_vs_erm']*100:>+12.1f}%"
            f"{r['source_domain_separability']['accuracy']*100:>8.1f}%"
            f"{r['delta_sharp']:>+12.4f}"
        )
    print("=" * 116)
    print(f"\nFull results (including per-class breakdowns) written to {out_path}")


if __name__ == "__main__":
    main()
