"""
Step 6: Controlled Design Study -- vary DANN's maximum gradient-reversal
strength over {0.25, 0.5, 1}, retaining the same schedule shape and every
other setting (lr, weight decay, seed, batch composition, domain_loss_weight,
discriminator architecture). Only grl_max_alpha changes between the three runs.

BEFORE running this: train the two new sweep points --
    python -m task2.train --config task2/configs/dann_alpha025.yaml
    python -m task2.train --config task2/configs/dann_alpha050.yaml
The alpha=1.0 point reuses the existing main-comparison DANN checkpoint
(task2/configs/dann.yaml -> checkpoints/dann_best.pt) rather than retraining
it -- same seed and fixed hyperparameters, so it IS that setting; retraining
it would just spend time reproducing (approximately -- MPS isn't perfectly
deterministic) the same result.

Then: python -m task2.evaluate_controlled_study

Per the spec: these target numbers are for ANALYSIS only. Do not go back
and use them to change grl_max_alpha, or anything else, for the Step 5
main-comparison DANN run -- that stays fixed at grl_max_alpha=1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.pacs import get_backbone_transforms
from common.pacs_protocol import build_or_load_source_splits, build_or_load_target_pool
from task2.evaluate_final import evaluate_one_method
from task2.train import get_device, load_config

# (grl_max_alpha, config_path, checkpoint/results run_name)
SWEEP_POINTS = [
    (0.25, "task2/configs/dann_alpha025.yaml", "dann_alpha025"),
    (0.50, "task2/configs/dann_alpha050.yaml", "dann_alpha050"),
    (1.00, "task2/configs/dann.yaml", "dann"),
]


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    base_cfg = load_config("task2/configs/base.yaml")
    source_splits = build_or_load_source_splits(base_cfg["data"]["pacs_root"])
    target_samples = build_or_load_target_pool(base_cfg["data"]["pacs_root"])
    eval_transform = get_backbone_transforms("val", base_cfg["data"]["resize"], base_cfg["data"]["crop"])

    results = {}
    for max_alpha, config_path, run_name in SWEEP_POINTS:
        print(f"\n=== grl_max_alpha={max_alpha} ({run_name}) ===")
        cfg = load_config(config_path)
        results[run_name] = {
            "grl_max_alpha": max_alpha,
            **evaluate_one_method(
                "dann", cfg, device, source_splits, target_samples, eval_transform, ckpt_name=run_name
            ),
        }

    results_dir = Path(base_cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "dann_controlled_study_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'grl_max_alpha':<16}{'Src Acc':>10}{'Src F1':>9}{'Tgt Acc':>10}{'Tgt F1':>9}{'DomSep':>10}")
    print("-" * 80)
    for run_name, r in results.items():
        print(
            f"{r['grl_max_alpha']:<16}"
            f"{r['mean_source_accuracy']*100:>9.1f}%"
            f"{r['mean_source_macro_f1']:>9.3f}"
            f"{r['target_accuracy']*100:>9.1f}%"
            f"{r['target_macro_f1']:>9.3f}"
            f"{r['domain_separability']*100:>9.1f}%"
        )
    print("=" * 80)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
