"""
Section 5: Controlled Design Study -- vary DAN-DG's lambda_dg over
{0.1, 1, 10}, keeping every other setting fixed (lr, weight decay, seed,
batch composition, kernel bandwidth multipliers). Only lambda_dg changes
between the three runs.

Expected effect (stated up front, per spec, BEFORE interpreting the
results -- the main-comparison run at lambda_dg=1 already showed a
collapse: train_cls_loss stuck at ln(7) the entire run while
train_mmd_loss decayed to ~0, source accuracy 21.7%, mean_val_macro_f1
frozen every epoch):
  - lambda_dg=0.1 (much weaker alignment pressure): source performance
    should recover toward ERM's ~94%, since classification can finally
    get real gradient signal; the diagnostic (source-domain separability)
    should stay HIGHER than at lambda_dg=1 (less alignment pressure ->
    less invariance); Sketch performance should improve over the
    collapsed lambda_dg=1 run but may still trail ERM if 0.1 isn't
    enough alignment to help generalization.
  - lambda_dg=10 (much stronger): expect the SAME collapse pattern as
    lambda_dg=1, likely even more severe/immediate -- source accuracy at
    or below chance, source-domain separability pushed further toward
    (or even below, in a degenerate collapse) the 33.3% chance floor,
    Sketch performance similarly collapsed.
  - lambda_dg=1 (already run, the main comparison): the collapsed point
    described above -- included here again via ckpt_name="dan_dg" so all
    three sweep points land in one results file/table.

The main Section-4 comparison stays fixed at lambda_dg=1 regardless of
what this sweep shows -- do not retroactively swap in a different
lambda_dg as if it had been the original choice.

BEFORE running this: train the two new sweep points --
    python -m task3.train --config task3/configs/dan_dg_lambda01.yaml
    python -m task3.train --config task3/configs/dan_dg_lambda10.yaml
The lambda_dg=1.0 point reuses the existing main-comparison checkpoint
(task3/configs/dan_dg.yaml -> checkpoints/dan_dg_best.pt) rather than
retraining it.

Then: python -m task3.evaluate_controlled_study
"""

from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn

from common.pacs import get_backbone_transforms
from common.pacs_protocol import build_or_load_source_splits, build_or_load_target_pool
from task3.evaluate_sketch import evaluate_one_method
from task3.evaluation.sharpness import build_fixed_sharpness_batch
from task3.train import get_device, load_config

# (lambda_dg, config_path, checkpoint/results run_name)
SWEEP_POINTS = [
    (0.1, "task3/configs/dan_dg_lambda01.yaml", "dan_dg_lambda01"),
    (1.0, "task3/configs/dan_dg.yaml", "dan_dg"),
    (10.0, "task3/configs/dan_dg_lambda10.yaml", "dan_dg_lambda10"),
]


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    base_cfg = load_config("task3/configs/base.yaml")
    source_splits = build_or_load_source_splits(base_cfg["data"]["pacs_root"])
    target_samples = build_or_load_target_pool(base_cfg["data"]["pacs_root"])
    eval_transform = get_backbone_transforms("val", base_cfg["data"]["resize"], base_cfg["data"]["crop"])
    criterion = nn.CrossEntropyLoss()

    # Same fixed batch (seed 6304, 32/source domain) used by evaluate_sketch.py's
    # main comparison, reused here so the sharpness numbers are comparable
    # across BOTH the method comparison and this sweep.
    fixed_sharpness_batch = build_fixed_sharpness_batch(
        source_splits, eval_transform, device, per_domain=32, seed=6304
    )

    results = {}
    for lambda_dg, config_path, run_name in SWEEP_POINTS:
        print(f"\n=== lambda_dg={lambda_dg} ({run_name}) ===")
        cfg = load_config(config_path)
        results[run_name] = {
            "lambda_dg": lambda_dg,
            **evaluate_one_method(
                "dan_dg", cfg, device, source_splits, target_samples, eval_transform,
                fixed_sharpness_batch, criterion, ckpt_name=run_name,
            ),
        }

    results_dir = Path(base_cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "dan_dg_controlled_study_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 96)
    print(
        f"{'lambda_dg':<12}{'SrcAcc(mean)':>13}{'SrcF1(mean)':>13}"
        f"{'SketchAcc':>11}{'SketchF1':>10}{'DomSep':>9}{'DeltaSharp':>12}"
    )
    print("-" * 96)
    for run_name, r in results.items():
        print(
            f"{r['lambda_dg']:<12}"
            f"{r['mean_source_accuracy']*100:>12.1f}%"
            f"{r['mean_source_macro_f1']:>13.3f}"
            f"{r['sketch_accuracy']*100:>10.1f}%"
            f"{r['sketch_macro_f1']:>10.3f}"
            f"{r['source_domain_separability']['accuracy']*100:>8.1f}%"
            f"{r['delta_sharp']:>+12.4f}"
        )
    print("=" * 96)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
