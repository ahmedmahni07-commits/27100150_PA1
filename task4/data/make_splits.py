"""
Standalone helper: python -m task4.data.make_splits
Pre-builds and caches the CIFAR-10 90/10 train/val split (seed 6304), and
downloads CIFAR-100's test partition, so every later run (vanilla/gcsc/
proser training, extract_outputs.py) sees the identical split without
recomputing it and without triggering a download mid-training-run.
"""

from __future__ import annotations

from task4.data.cifar10 import build_or_load_train_val_split
from task4.data.cifar100_unknowns import build_unknown_datasets, NEAR_UNKNOWN_CLASSES, FAR_UNKNOWN_CLASSES

DATA_ROOT = "task4/data/raw"


def main() -> None:
    train_idx, val_idx = build_or_load_train_val_split(DATA_ROOT)
    print(f"CIFAR-10: train={len(train_idx)} val={len(val_idx)} (seed 6304, 90/10 stratified)")

    from torchvision import transforms
    near, far = build_unknown_datasets(DATA_ROOT, transforms.ToTensor())
    print(f"CIFAR-100 near-unknown ({len(NEAR_UNKNOWN_CLASSES)} classes): {len(near)} images")
    print(f"CIFAR-100 far-unknown ({len(FAR_UNKNOWN_CLASSES)} classes): {len(far)} images")


if __name__ == "__main__":
    main()
