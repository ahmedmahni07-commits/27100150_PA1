"""
CIFAR-10 known-class dataset loading, plus the required stratified 90/10
split of the official CIFAR-10 TRAINING partition (seed 6304). Only the
training portion is used for optimization; the official CIFAR-10 TEST set
is used ONLY for final known-class evaluation (CSA), never for training
or checkpoint selection.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

SEED = 6304
VAL_FRACTION = 0.1

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(split: str) -> transforms.Compose:
    """
    Per spec, unless a step explicitly changes it: random crop to 32x32
    with 4-pixel padding + random horizontal flip for training; plain
    normalization (no augmentation) for val/test/eval.
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if split == "train":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([transforms.ToTensor(), normalize])


def get_gcsc_train_transform() -> transforms.Compose:
    """
    GCSC's ONE change from the vanilla recipe: RandAugment(num_ops=2,
    magnitude=9) inserted after crop+flip and before ToTensor/normalize.
    Everything else (optimizer, schedule, batch size, epochs, seed,
    checkpoint rule) stays identical to vanilla -- see task4/methods/gcsc.py.
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
    ])


class TransformedSubset(Dataset):
    """
    Wraps a base dataset (built with transform=None) plus an explicit
    index list and its own transform, so train/val can share ONE
    downloaded CIFAR-10 training-partition object without leaking indices
    or forcing both splits to use the same transform.
    """

    def __init__(self, base_dataset, indices: list[int], transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.base_dataset[self.indices[i]]  # PIL image, int label
        return self.transform(img), label


def build_or_load_train_val_split(
    data_root: str | Path,
    cache_path: str | Path = "task4/cache/cifar10_train_val_split_seed6304.json",
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[int], list[int]]:
    """
    Stratified 90/10 split of the official CIFAR-10 training partition's
    50000 indices, seeded 6304, cached so every run/method (vanilla,
    gcsc, proser, extract_outputs.py, ...) sees the identical split.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        return raw["train_idx"], raw["val_idx"]

    base = CIFAR10(root=str(data_root), train=True, download=True)
    labels = base.targets  # list[int], avoids decoding every image just for labels
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, stratify=labels, random_state=seed,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx}, f)

    return train_idx, val_idx


def build_datasets(data_root: str | Path, train_transform, eval_transform):
    """
    Returns (train_dataset, val_dataset, test_dataset). train/val share
    ONE underlying download of the CIFAR-10 training partition (base
    dataset created with transform=None; each subset applies its own
    transform); test is the official CIFAR-10 test set (eval_transform,
    used only for final known-class evaluation / CSA).
    """
    train_idx, val_idx = build_or_load_train_val_split(data_root)

    base_train = CIFAR10(root=str(data_root), train=True, download=True, transform=None)
    train_dataset = TransformedSubset(base_train, train_idx, train_transform)
    val_dataset = TransformedSubset(base_train, val_idx, eval_transform)

    test_dataset = CIFAR10(root=str(data_root), train=False, download=True, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset
