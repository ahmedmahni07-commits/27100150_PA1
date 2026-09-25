"""
Entry point: python -m task2.evaluate_final

Run ONLY after every method's checkpoints/configs are frozen. Source-only's
own target numbers were already reported by its own train.py run (Step 1
explicitly permits that early peek); this script is the first and only
place DAN's, DANN's, and CDAN's target labels get used.

Produces (Step 5's Required Evidence):
  - source-val + target accuracy/macro-F1 for Source-only, DAN, DANN, CDAN
  - target accuracy change relative to Source-only
  - domain separability score per method
  - per-class target accuracy + biggest improvements/degradations vs Source-only
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.pacs import PACSDataset, PACSSample, get_backbone_transforms
from common.pacs_protocol import (
    build_or_load_source_splits,
    build_or_load_target_pool,
    make_source_eval_loader,
    make_target_eval_loader,
)
from task2.evaluation.class_analysis import (
    compare_to_baseline,
    confusion_matrix_for_class,
    per_class_accuracy,
)
from task2.evaluation.domain_separability import compute_domain_separability
from task2.evaluation.metrics import evaluate_loader
from task2.methods import cdan, dan, dann, source_only
from task2.train import get_device, load_config

METHOD_MODULES = {
    "source_only": source_only,
    "dan": dan,
    "dann": dann,
    "cdan": cdan,
}

METHOD_CONFIGS = {
    "source_only": "task2/configs/source_only.yaml",
    "dan": "task2/configs/dan.yaml",
    "dann": "task2/configs/dann.yaml",
    "cdan": "task2/configs/cdan.yaml",
}


@torch.no_grad()
def collect_features(model, loader, device: torch.device) -> np.ndarray:
    """Run `model.backbone` over `loader`, return stacked features as a numpy array."""
    model.eval_mode()
    all_feats = []
    for images, _labels, _domains in loader:
        images = images.to(device, non_blocking=True)
        all_feats.append(model.backbone(images).cpu().numpy())
    return np.concatenate(all_feats)


def evaluate_one_method(
    method: str,
    cfg: dict,
    device: torch.device,
    source_splits: dict,
    target_samples: list,
    eval_transform,
    ckpt_name: str | None = None,
) -> dict:
    """
    `ckpt_name` overrides which checkpoint file to load (defaults to
    `method`) -- used by evaluate_controlled_study.py so a sweep over
    e.g. dann_alpha025/dann_alpha050 checkpoints doesn't require a
    separate copy of this whole function.
    """
    method_module = METHOD_MODULES[method]
    model = method_module.build_model(cfg, device)

    ckpt_name = ckpt_name or method
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_path = checkpoint_dir / f"{ckpt_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} -- run "
            f"`python -m task2.train --config <its config>` first."
        )
    method_module.load_checkpoint(model, str(ckpt_path), device)

    # Source-validation performance, per domain and pooled.
    per_domain_val = {}
    for domain, split in source_splits.items():
        loader = make_source_eval_loader(split, eval_transform)
        result = evaluate_loader(model, loader, device)
        per_domain_val[domain] = {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
    mean_source_acc = float(np.mean([v["accuracy"] for v in per_domain_val.values()]))
    mean_source_f1 = float(np.mean([v["macro_f1"] for v in per_domain_val.values()]))

    # Target performance. Labels are read here for the first time for
    # DAN/DANN/CDAN (source-only's target numbers were already reported
    # during its own training run, per Step 1's explicit carve-out).
    target_loader = make_target_eval_loader(target_samples, eval_transform)
    target_result = evaluate_loader(model, target_loader, device)

    # Domain separability: pool ALL source-val samples (not just one
    # domain) against the full target set, equal counts, 70/30, seed 6304.
    pooled_source_samples = [
        PACSSample(path=p, label=l, domain=domain)
        for domain, split in source_splits.items()
        for p, l in zip(split.val_paths, split.val_labels)
    ]
    pooled_source_loader = DataLoader(
        PACSDataset(pooled_source_samples, eval_transform), batch_size=64, shuffle=False, num_workers=2
    )
    source_feats = collect_features(model, pooled_source_loader, device)
    target_feats = collect_features(model, target_loader, device)
    domain_sep = compute_domain_separability(source_feats, target_feats)

    per_class_acc = per_class_accuracy(
        target_result["y_true"], target_result["y_pred"], num_classes=cfg["model"]["num_classes"]
    )

    return {
        "per_domain_val": per_domain_val,
        "mean_source_accuracy": mean_source_acc,
        "mean_source_macro_f1": mean_source_f1,
        "target_accuracy": target_result["accuracy"],
        "target_macro_f1": target_result["macro_f1"],
        "domain_separability": domain_sep,
        "per_class_target_accuracy": per_class_acc.tolist(),
        # Raw target predictions, kept so confusion matrices (and anything else)
        # can be computed later without a second forward pass through the model.
        "target_y_true": target_result["y_true"].tolist(),
        "target_y_pred": target_result["y_pred"].tolist(),
    }


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    base_cfg = load_config("task2/configs/base.yaml")
    source_splits = build_or_load_source_splits(base_cfg["data"]["pacs_root"])
    target_samples = build_or_load_target_pool(base_cfg["data"]["pacs_root"])
    eval_transform = get_backbone_transforms("val", base_cfg["data"]["resize"], base_cfg["data"]["crop"])

    results = {}
    for method, config_path in METHOD_CONFIGS.items():
        print(f"\n=== Evaluating {method} ===")
        cfg = load_config(config_path)
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

            # Required Evidence: "selected confusions or failure cases that support
            # a claim about positive or negative transfer." For every flagged class
            # (most improved + most degraded vs Source-only), record what it's
            # actually predicted as -- for Source-only and for this method -- so a
            # before/after comparison is directly visible, not just the accuracy delta.
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

    results_dir = Path(base_cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluate_final_results.json"
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
    print(f"\nFull results (including per-class breakdowns) written to {out_path}")


if __name__ == "__main__":
    main()
