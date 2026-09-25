# ATML PA1 — Beyond IID (27100150, Ahmed Waqas)

Four controlled studies of what happens when the IID assumption breaks:
**Task 1** inductive biases (STL-10; ResNet-50 / ViT-B/16 / CLIP), **Task 2** unsupervised
domain adaptation (PACS, Sketch target), **Task 3** domain generalization (Sketch unseen),
**Task 4** open-set recognition (CIFAR-10 known, CIFAR-100 unknown).
The report is in `report/PA1_report_27100150.pdf` (source: `report/main.tex`).

All commands are run **from the repository root**. Seed 6304 everywhere.

## Environment

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git submodule update --init          # task1/pytorch-AdaIN (naoto0804/pytorch-AdaIN)
```
Runs used Apple-silicon MPS. PROSER's manifold mixup samples Beta(2,2) on the CPU
(`aten::_sample_dirichlet` is not implemented on MPS).

## Data (not committed)

| Dataset | Where it goes | How |
|---|---|---|
| STL-10 | `task1/data/raw/stl10_binary/` | downloaded by `task1/data/make_subset.py` |
| PACS | `common/pacs/images/<domain>/<class>/*.jpg` | download PACS, unzip there |
| CIFAR-10 / CIFAR-100 | `task4/data/raw/` | downloaded by torchvision on first run |

Committed split files: `task1/data/splits/stl10_splits.pt` (train/val indices + 500-image test subset),
`common/splits/pacs_source_splits_seed6304.json`, `common/splits/pacs_sketch_seed6304.json`,
`task4/cache/cifar10_train_val_split_seed6304.json`.

## Task 1 — inductive biases

```bash
python task1/data/make_subset.py               # splits + 500-image test subset
python task1/analysis/extract_features.py      # cached train/val/test features -> task1/features/
python task1/analysis/texture_shape.py         # content/style pairs -> task1/data/cue_conflicts_raw/
python task1/data/generate_adain.py            # AdaIN stylisation -> task1/data/cue_conflicts_stylized/
python task1/data/generate_patch_shuffle.py    # fixed 4x4 shuffles -> task1/data/patch_shuffle/
python task1/scripts/run_task1.py              # ALL Task 1 numbers + figures -> task1/results/
```
`run_task1.py` trains the three linear heads, evaluates clean / grayscale / hue / translation /
patch-shuffle / cue-conflict, representation stability and t-SNE, and writes
`task1/results/task1_results.json`, `cue_conflict_predictions.csv` and the figures.
The older `task1/analysis/*` evaluation scripts are superseded by it.
Note: the stylised set used in the report has 325 images (22–44 per direction) because two
generation passes wrote overlapping indices; the exact file list is in `cue_conflict_predictions.csv`.

## Task 2 — UDA (PACS)

```bash
for m in source_only dan dann cdan dann_alpha025 dann_alpha050; do
  python -m task2.train --config task2/configs/$m.yaml
done
python -m task2.evaluate_final               # main table, separability, per-class, confusions
python -m task2.evaluate_controlled_study    # DANN max-GRL sweep {0.25, 0.5, 1}
python -m task2.evaluation.plot_confusion_matrices
```
Sketch labels are read only by the two evaluation scripts. **Deviation:** DANN/CDAN diverged when
implemented exactly as specified; they use gradient clipping (norm 5) and an L2-normalised
discriminator input (see `task2/DEVIATIONS.md`; pre-fix runs kept in `task2/results/pre_stabilization/`).

## Task 3 — DG (Sketch unseen)

```bash
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
python -m task3.train --config task3/configs/dan_dg_lambda01.yaml
python -m task3.train --config task3/configs/dan_dg_lambda10.yaml
python -m task3.evaluate_sketch              # the only script that loads Sketch
python -m task3.evaluate_controlled_study    # DAN-DG lambda sweep {0.1, 1, 10}
```
ERM is the Task 2 source-only checkpoint (`task3/configs/erm.yaml` points to it; not retrained).

## Task 4 — OSR (CIFAR)

```bash
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml      # starts from vanilla_best.pt
for m in vanilla gcsc proser; do python -m task4.extract_outputs --method $m; done
python -m task4.evaluate_osr                        # Tables 1-2, score figure, failure cases
python -m task4.evaluation.mahalanobis_diag         # spec-compliant diagonal-covariance Mahalanobis
python -m task4.evaluation.per_class_acceptance     # per-class acceptance + score agreement
python -m task4.evaluation.mls_acceptance           # test acceptance / near-far rejection per model
```
CIFAR-100 is read only by `extract_outputs` / evaluation, after all checkpoints and thresholds are fixed.
`scores/mahalanobis.py` uses a full shared covariance; the report's main Mahalanobis row is the
diagonal version from `evaluation/mahalanobis_diag.py`. RPL (optional) was not implemented.

## Results

Every number in the report is in a JSON/CSV under `task*/results/`
(checkpoints and cached features/logits are git-ignored).

## Attribution

AdaIN stylisation: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (code and
pretrained VGG/decoder weights). Backbones: torchvision and open_clip. MMD, gradient reversal,
DANN/CDAN discriminators, SAM, PROSER and all OSR scores are implemented in this repo from the
cited papers.
