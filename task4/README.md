# Task 4 — Open-Set Recognition (OSR)

CIFAR-10 = ten known classes. CIFAR-100 (fixed near/far fine-class
groups, TEST split only, 800 images/group) is evaluation-only: it may
never influence training, checkpoint selection, score design, or
threshold selection.

## What's built

**Data / models**
- `data/cifar10.py` — stratified 90/10 train/val split of the official
  CIFAR-10 training partition (seed 6304, cached), test set held back for
  final CSA. `get_transforms("train"/"eval")` (crop+flip vs. plain
  normalize) and `get_gcsc_train_transform()` (+RandAugment).
- `data/cifar100_unknowns.py` — near/far unknown pools from CIFAR-100's
  TEST split only, filtered to the fixed 8-class groups.
- `data/make_splits.py` — standalone: `python -m task4.data.make_splits`
- `models/resnet_cifar.py` — CIFAR ResNet-18 (3x3 stride-1 stem, no
  initial maxpool, random init), with `forward_pre_layer3`/
  `forward_post_layer3` split points for PROSER's manifold mixup.
- `models/classifier_head.py` — 512 → 10 linear head.

**Step 1 — Vanilla**: `methods/vanilla.py`, `configs/vanilla.yaml`

**Step 2 — Post-hoc scores** (all read from `extract_outputs.py`'s cache,
so every score sees identical saved logits/features):
`scores/msp.py`, `scores/mls.py`, `scores/energy.py`,
`scores/mahalanobis.py` (shared pooled covariance + 1e-6 diagonal ridge,
fit on unaugmented CIFAR-10 train features only)

**Step 3 — GCSC**: `methods/gcsc.py` re-exports vanilla's model/training
unchanged — only `train.py`'s transform choice differs. `configs/gcsc.yaml`

**Step 4 — PROSER**: `methods/manifold_mixup.py` (different-class pairing
+ per-pair Beta(2,2) mixing at the layer2/layer3 boundary),
`methods/proser.py` (5 dummy classifiers collapsed via max-logit into one
placeholder response; classifier-placeholder loss β=1 + data-placeholder
loss γ=0.1 from Zhou et al. 2021, verified against arXiv 2103.15086;
single backward per step, batch split in half as specified).
`configs/proser.yaml` (its own lr=1e-3/50-epoch recipe, initializes from
`vanilla_best.pt`).

**Pipeline**
- `train.py` — SGD lr=0.1/momentum=0.9/wd=5e-4 cosine decay, batch 128,
  100 epochs, seed 6304 (PROSER's config overrides lr/epochs), selects by
  CIFAR-10 **validation accuracy**, reports final **test** accuracy.
  Handles vanilla/gcsc/proser uniformly via method dispatch.
- `extract_outputs.py` — freezes a checkpoint, runs it once over CIFAR-10
  train(unaugmented)/val/test + CIFAR-100 near/far, caches features +
  logits (+ dummy logits for PROSER) to `cache/<method>_outputs.npz`.
- `evaluate_osr.py` — Required Evidence: Table 1 (MSP/MLS/Energy/
  Mahalanobis on frozen Vanilla — near/far/all AUROC + 95th-percentile
  validation-calibrated acceptance/FPR@95TPR), Table 2 (Vanilla/GCSC/
  PROSER comparison, CSA + OSR metrics, MLS common score + PROSER's own
  placeholder-score row), a 3-panel MSP/MLS/Mahalanobis score-distribution
  figure, and 3 near + 3 far incorrectly-accepted failure examples
  (vanilla MLS threshold).
- `evaluation/metrics.py`, `evaluation/thresholds.py`,
  `evaluation/failure_analysis.py` — shared building blocks above.

## Not built

- `methods/rpl.py` (Step 5, **optional** extension) — Reciprocal Point
  Learning; not started.

## How to run

```
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml   # after vanilla finishes -- initializes from vanilla_best.pt

python -m task4.extract_outputs --method vanilla
python -m task4.extract_outputs --method gcsc
python -m task4.extract_outputs --method proser

python -m task4.evaluate_osr
```

(First run downloads CIFAR-10/CIFAR-100 via torchvision into
`task4/data/raw/` — needs network access.)
