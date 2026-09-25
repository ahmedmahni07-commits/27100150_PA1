# Task 2 — Documented Deviations from the Assignment Manual

## Problem observed

Across all three DANN configurations (`grl_max_alpha` in {0.25, 0.5, 1.0}) and
under a plain, un-clipped, un-normalized implementation exactly matching the
manual's specification (discriminator: 256-unit hidden layer, ReLU, dropout
0.5, 2-class output; GRL schedule alpha(p) = max_alpha * (2/(1+exp(-10p)) - 1);
domain loss weighted 1:1 against classification loss; AdamW lr=1e-4,
wd=1e-4), training diverged numerically:

- `grl_max_alpha=1.0`: `train_domain_loss` was already 2968.7 at epoch 0
  (mean source-val macro-F1 0.073, below the 1/7 chance rate) and grew to
  order 1e8-1e9 by epoch 7-8.
- `grl_max_alpha=0.25` and `0.5`: a clean epoch 0 (val macro-F1 0.90 and 0.81
  respectively), then `train_domain_loss` exploded by 3-4 orders of magnitude
  at epoch 1 and never recovered for the remainder of the run.
- In every case, the post-hoc domain-separability probe (Common Evaluation
  step) still reported ~99.6-99.7% separability -- essentially unchanged from
  Source-only's 99.9% -- despite training-time domain-discriminator accuracy
  sitting near chance (~50%) for most of these runs. That combination
  (near-chance discriminator accuracy during training, near-ceiling
  separability from an independently trained probe afterward, total
  classification collapse) indicates the shared representation collapsed
  rather than became genuinely domain-invariant.

CDAN did not diverge as catastrophically, but showed a comparable, milder
instability signature (a sharp mid-training dip in val macro-F1 with a
train-loss spike around epoch 4, partially recovering by epoch 6).

## Fixes applied

Both changes are training-stability techniques only. No specified
hyperparameter (learning rate, weight decay, `domain_loss_weight`, GRL
schedule shape, batch composition, optimizer choice, epoch budget, early
stopping rule, discriminator architecture) was changed. `source_only.yaml`
and `dan.yaml` are untouched -- their training was already stable and their
previously reported results remain valid and reproducible exactly as before.

1. **Gradient-norm clipping** (`task2/train.py`, gated by a new
   `optim.grad_clip_norm` config field, `null` by default). When set,
   `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=...)` is
   applied after `training_step()`'s `backward()` call and before
   `optimizer.step()`. Set to `5.0` in `dann.yaml`, `dann_alpha025.yaml`,
   `dann_alpha050.yaml`, and `cdan.yaml` only. The realized gradient norm is
   now also logged per step (`grad_norm` -> averaged into `train_grad_norm`
   in the results JSON/history), so the clipping's effect is directly
   visible in the training curves.

2. **L2-normalization of the domain discriminator's input**
   (`task2/models/domain_discriminator.py`:
   `normalize_discriminator_input`, applied in `task2/methods/dann.py` and
   `task2/methods/cdan.py`). Each row of the feature fed to the GRL +
   discriminator is L2-normalized -- the raw 512-d feature for DANN, the
   flattened `f (x) p` outer product for CDAN -- bounding the
   discriminator's input scale (and therefore the domain loss/gradient
   magnitude the GRL reverses) regardless of the backbone's raw activation
   magnitude. This is applied **only** on the domain-discriminator branch:
   the features/probabilities used for the classification loss and head are
   untouched and remain un-normalized, un-detached, exactly as the manual
   specifies (CDAN's `use_entropy_conditioning`, `detach_features`, and
   `detach_probs` all remain `false`, per spec).

## What still needs to happen

`dann.yaml`, `dann_alpha025.yaml`, `dann_alpha050.yaml`, and `cdan.yaml` all
need to be re-trained with these changes; the checkpoints/results currently
in `task2/results/` predate this fix and reflect the un-clipped,
un-normalized runs. Recommended: keep the original result JSONs (rename or
move them, e.g. into `task2/results/pre_stabilization/`) so the before/after
comparison itself can be reported as evidence that the fix worked, rather
than overwriting them outright.
