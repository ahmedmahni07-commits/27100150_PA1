"""
Shared PACS dataset protocol for Tasks 2 and 3.

Defines the domain/class vocabulary, reads the official PACS split files,
and builds the deterministic stratified 80/20 source train/val split
(seed 6304) that both tasks must reuse unchanged (per the assignment:
"Reuse the same source splits across both tasks").

Design notes (why it's built this way):
- PACS *_train_kfold.txt is the "official training partition" for a domain.
  For the three SOURCE domains (Photo, Art Painting, Cartoon) we take this
  file as the pool and carve our OWN stratified 80/20 train/val split out of
  it with seed 6304 -- the assignment is explicit that split is ours to make,
  not the official crossval/test files.
- For the TARGET domain (Sketch), the assignment says "the complete target
  domain serves as the unlabeled adaptation set" -- so we use every Sketch
  image (train+crossval+test combined), since Sketch is never split into
  train/val the way source domains are.
- The chosen source split is cached to disk (pacs_source_splits_seed6304.json)
  the first time it's computed, so Task 2 and Task 3 -- which import this
  same module -- are guaranteed to reuse the identical split rather than
  each independently reconstructing one that happens to match.
"""
import json
import os
from collections import defaultdict

import numpy as np

# PACS kfold split files use 1-indexed labels in alphabetical class order,
# e.g. "art_painting/dog/pic_333.jpg 1" -> dog. We convert to 0-indexed here
# so every downstream consumer (loss functions, sklearn, etc.) gets plain
# 0..6 class ids and never has to remember the off-by-one.
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
SOURCE_DOMAIN_TO_IDX = {d: i for i, d in enumerate(SOURCE_DOMAINS)}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.join(_THIS_DIR, "pacs", "images")
SPLITS_ROOT = os.path.join(_THIS_DIR, "pacs", "splits")

SEED = 6304
VAL_FRACTION = 0.2

_CACHE_PATH = os.path.join(_THIS_DIR, f"pacs_source_splits_seed{SEED}.json")


def _read_kfold_file(path):
    """Read one PACS *_kfold.txt file -> list of (relpath, 0-indexed label)."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath, label_1indexed = line.rsplit(" ", 1)
            samples.append((relpath, int(label_1indexed) - 1))
    return samples


def official_train_pool(domain):
    """The 'official training partition' for a domain (used for source domains)."""
    path = os.path.join(SPLITS_ROOT, f"{domain}_train_kfold.txt")
    return _read_kfold_file(path)


def full_domain_pool(domain):
    """Every official image for a domain (train+crossval+test, deduplicated).
    Used to build the complete, unlabeled target (Sketch) pool."""
    seen = {}
    for split_name in ("train_kfold", "crossval_kfold", "test_kfold"):
        path = os.path.join(SPLITS_ROOT, f"{domain}_{split_name}.txt")
        for relpath, label in _read_kfold_file(path):
            seen[relpath] = label
    return sorted(seen.items())


def _stratified_split(samples, val_fraction, seed):
    """Deterministic per-class stratified split into (train, val)."""
    by_class = defaultdict(list)
    for relpath, label in samples:
        by_class[label].append(relpath)

    train, val = [], []
    for label in sorted(by_class):
        paths = sorted(by_class[label])       # sort before shuffling: deterministic input order
        rng = np.random.RandomState(seed)     # fresh RNG per class: split is independent of
        rng.shuffle(paths)                    # domain/class iteration order
        n_val = max(1, round(len(paths) * val_fraction))
        val.extend((p, label) for p in paths[:n_val])
        train.extend((p, label) for p in paths[n_val:])
    return sorted(train), sorted(val)


def build_source_splits(force_recompute=False):
    """Return {domain: {'train': [(relpath,label),...], 'val': [...]}} for the
    three PACS source domains. Cached to disk after first computation."""
    if os.path.exists(_CACHE_PATH) and not force_recompute:
        with open(_CACHE_PATH, "r") as f:
            raw = json.load(f)
        return {
            domain: {
                "train": [tuple(x) for x in split["train"]],
                "val": [tuple(x) for x in split["val"]],
            }
            for domain, split in raw.items()
        }

    splits = {}
    for domain in SOURCE_DOMAINS:
        pool = official_train_pool(domain)
        train, val = _stratified_split(pool, VAL_FRACTION, SEED)
        splits[domain] = {"train": train, "val": val}

    with open(_CACHE_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def target_pool():
    """The complete, unlabeled-during-training Sketch domain."""
    return full_domain_pool(TARGET_DOMAIN)


if __name__ == "__main__":
    splits = build_source_splits()
    total_train = total_val = 0
    for domain, split in splits.items():
        n_tr, n_val = len(split["train"]), len(split["val"])
        total_train += n_tr
        total_val += n_val
        print(f"{domain:>14}: train={n_tr:5d}  val={n_val:5d}")
    print(f"{'TOTAL source':>14}: train={total_train:5d}  val={total_val:5d}")
    print(f"{'sketch (target)':>14}: {len(target_pool())} images (unlabeled during training)")
