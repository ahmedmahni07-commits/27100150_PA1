# Task 3 — Domain Generalization

Protocol: Sketch is NEVER loaded during training, source-side diagnostics,
or checkpoint/hyperparameter selection anywhere in this directory — only
`evaluate_sketch.py` touches it, at the very end. `train.py` only imports
`common.pacs_protocol.build_or_load_source_splits` / `source_balanced_batches`
/ `steps_per_epoch_source_only`; it never imports `build_or_load_target_pool`.

## Methods

- **ERM** (`methods/erm.py`, `configs/erm.yaml`) — Task 2's source-only
  checkpoint (`task2/results/checkpoints/source_only_best.pt`), reused
  unchanged. Not retrained here.
- **DAN-DG** (`methods/dan_dg.py`, `configs/dan_dg.yaml`) — pairwise MMD
  alignment across the three source domains (photo/art/cartoon), no target
  term. `L = L_cls + lambda_dg * mean_pairs MMD^2(f_i, f_j)`.
- **SAM** (`methods/sam.py`, `configs/sam.yaml`) — Sharpness-Aware
  Minimization on the plain ERM loss over domain-balanced source batches.
  Two-pass ascent/descent update, own loop in `train.py` (`SAMOptimizer`).

All three share the same backbone/head architecture, optimizer (AdamW,
lr=1e-4, wd=1e-4), 30-epoch budget, patience-5 early stopping, and seed
6304 — selection is always mean macro-F1 over the three source validation
splits (`selection/source_validation.py`).

## How to run

```
python -m task3.train --config task3/configs/erm.yaml
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
```

Then, once all three checkpoints exist:

```
python -m task3.evaluate_sketch
```

(`evaluate_sketch.py` — TODO, mirrors `task2/evaluate_final.py`: loads all
three checkpoints, evaluates each on the full Sketch set for the first
time, builds the comparison table via
`evaluation/domain_metrics.build_comparison_table`, runs
`evaluation/source_domain_separability.py` per method, and runs
`evaluation/sharpness.py`'s fixed-batch Delta_sharp comparison for ERM vs.
SAM.)

## TODO after running

- exact hyperparameters used for the main comparison (lambda_dg, rho) and
  for each method's controlled design study (see `controlled_study` blocks
  in `configs/dan_dg.yaml` / `configs/sam.yaml`)
- final Required Evidence numbers and writeup
