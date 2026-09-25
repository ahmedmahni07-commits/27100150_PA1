"""
The single shared PACS protocol used by Task 2 and Task 3:
  - seed 6304 everywhere
  - stratified 80/20 train/val split per source domain
  - domain-balanced batch construction (8 per source domain per step, 24 target)
  - the frozen Sketch split cache so Task 2 and Task 3 use identical data

Task-specific losses (DAN/DANN/CDAN in Task 2, DAN-DG/SAM in Task 3) must
NOT live here — only the data/splitting/loader machinery identical across
both tasks.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common.pacs import PACSSample, PACSDataset, SOURCE_DOMAINS, TARGET_DOMAIN, list_domain_images

SEED = 6304
VAL_FRACTION = 0.2
SOURCE_EXAMPLES_PER_DOMAIN_PER_BATCH = 8   # -> 24 source examples/batch
TARGET_EXAMPLES_PER_BATCH = 24             # equal total source/target


def set_all_seeds(seed: int = SEED) -> None:
    """Seed python/numpy/torch (CPU+CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DomainSplit:
    domain: str
    train_paths: list[str]
    train_labels: list[int]
    val_paths: list[str]
    val_labels: list[int]


def make_stratified_split(
    samples: list[PACSSample], seed: int = SEED, val_fraction: float = VAL_FRACTION
) -> tuple[list[PACSSample], list[PACSSample]]:
    """Stratified 80/20 split by class label, seeded with `seed`."""
    labels = [s.label for s in samples]
    indices = list(range(len(samples)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, stratify=labels, random_state=seed,
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def build_or_load_source_splits(
    pacs_root: str | Path,
    cache_path: str | Path = "common/splits/pacs_source_splits_seed6304.json",
) -> dict[str, DomainSplit]:
    """
    For each domain in SOURCE_DOMAINS, build (or load from cache) the
    stratified 80/20 split. Identical whether called from Task 2 or Task 3.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        return {
            domain: DomainSplit(**entry)  # entry already carries its own "domain" key
            for domain, entry in raw.items()
        }

    set_all_seeds(SEED)
    splits: dict[str, DomainSplit] = {}
    for domain in SOURCE_DOMAINS:
        samples = list_domain_images(pacs_root, domain)
        train_samples, val_samples = make_stratified_split(samples, seed=SEED, val_fraction=VAL_FRACTION)
        splits[domain] = DomainSplit(
            domain=domain,
            train_paths=[s.path for s in train_samples],
            train_labels=[s.label for s in train_samples],
            val_paths=[s.path for s in val_samples],
            val_labels=[s.label for s in val_samples],
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({d: asdict(s) for d, s in splits.items()}, f, indent=2)

    return splits


def build_or_load_target_pool(
    pacs_root: str | Path,
    cache_path: str | Path = "common/splits/pacs_sketch_seed6304.json",
) -> list[PACSSample]:
    """
    Load the complete Sketch domain as the unlabeled adaptation pool
    (Task 2) / eval-only pool (Task 3), caching the selected image
    identifiers so both tasks and reruns see the exact same set.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        return [
            PACSSample(path=p, label=l, domain=TARGET_DOMAIN)
            for p, l in zip(raw["paths"], raw["labels"])
        ]

    samples = sorted(list_domain_images(pacs_root, TARGET_DOMAIN), key=lambda s: s.path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {"paths": [s.path for s in samples], "labels": [s.label for s in samples]},
            f, indent=2,
        )

    return samples


class _CyclicIndexSampler:
    """
    Yields indices from range(n) in shuffled order, reshuffling (with a
    fresh seeded permutation) whenever exhausted, so callers can request
    an arbitrary number of indices without ever running out — this is what
    lets a shorter domain/pool "cycle" while a longer one is still being
    consumed within the same epoch.
    """

    def __init__(self, n: int, seed: int):
        self._n = n
        self._rng = np.random.RandomState(seed)
        self._order = self._rng.permutation(self._n)
        self._pos = 0

    def next(self, k: int) -> np.ndarray:
        out: list[int] = []
        while len(out) < k:
            remaining = self._n - self._pos
            take = min(k - len(out), remaining)
            out.extend(self._order[self._pos: self._pos + take].tolist())
            self._pos += take
            if self._pos >= self._n:
                self._order = self._rng.permutation(self._n)
                self._pos = 0
        return np.array(out)


def steps_per_epoch(
    source_splits: dict[str, DomainSplit],
    target_samples: list[PACSSample],
    source_per_domain: int = SOURCE_EXAMPLES_PER_DOMAIN_PER_BATCH,
    target_per_batch: int = TARGET_EXAMPLES_PER_BATCH,
) -> int:
    """
    One epoch = enough steps for every relevant pool (each source domain's
    train split, and the target pool) to be fully consumed at least once.
    Shorter pools cycle (reshuffling) to fill out the remaining steps.
    """
    src_steps = [
        -(-len(split.train_paths) // source_per_domain)  # ceil division
        for split in source_splits.values()
    ]
    tgt_steps = -(-len(target_samples) // target_per_batch)
    return max(*src_steps, tgt_steps)


def domain_balanced_batches(
    source_splits: dict[str, DomainSplit],
    target_samples: list[PACSSample],
    train_transform,
    source_per_domain: int = SOURCE_EXAMPLES_PER_DOMAIN_PER_BATCH,
    target_per_batch: int = TARGET_EXAMPLES_PER_BATCH,
    seed: int = SEED,
    epoch: int = 0,
):
    """
    Generator over one epoch's worth of batches, each with the fixed
    composition: `source_per_domain` examples from EVERY source domain
    (e.g. 8+8+8=24) plus `target_per_batch` target examples (24) — 48 total.

    Call once per training epoch with an incremented `epoch` so the
    per-domain shuffle order changes across epochs while staying fully
    determined by (seed, epoch) for reproducibility.

    Image decoding here is synchronous (no DataLoader workers) so the exact
    batch composition is trivial to reason about and debug; wrap this in a
    background thread/process later if throughput becomes a bottleneck.
    """
    domain_names = list(source_splits.keys())
    epoch_seed = seed + epoch

    samplers = {
        domain: _CyclicIndexSampler(len(split.train_paths), seed=epoch_seed + i)
        for i, (domain, split) in enumerate(source_splits.items())
    }
    target_sampler = _CyclicIndexSampler(len(target_samples), seed=epoch_seed + len(domain_names))

    n_steps = steps_per_epoch(source_splits, target_samples, source_per_domain, target_per_batch)

    for _ in range(n_steps):
        source_images, source_labels, source_domain_ids = [], [], []
        for domain_id, domain in enumerate(domain_names):
            split = source_splits[domain]
            idxs = samplers[domain].next(source_per_domain)
            for idx in idxs:
                img = Image.open(split.train_paths[idx]).convert("RGB")
                source_images.append(train_transform(img))
                source_labels.append(split.train_labels[idx])
                source_domain_ids.append(domain_id)

        target_images = []
        idxs = target_sampler.next(target_per_batch)
        for idx in idxs:
            img = Image.open(target_samples[idx].path).convert("RGB")
            target_images.append(train_transform(img))

        yield {
            "source_images": torch.stack(source_images),
            "source_labels": torch.tensor(source_labels, dtype=torch.long),
            "source_domain_ids": torch.tensor(source_domain_ids, dtype=torch.long),
            "target_images": torch.stack(target_images),
            "domain_names": domain_names,  # index -> name, for logging/debugging
        }


def steps_per_epoch_source_only(
    source_splits: dict[str, DomainSplit],
    source_per_domain: int = SOURCE_EXAMPLES_PER_DOMAIN_PER_BATCH,
) -> int:
    """
    Task 3 (Domain Generalization) analogue of steps_per_epoch: no target
    pool exists to fold into the max(), since Sketch is never loaded during
    Task 3 training. One epoch = enough steps for every source domain's
    train split to be fully consumed at least once (shorter domains cycle).
    """
    src_steps = [
        -(-len(split.train_paths) // source_per_domain)  # ceil division
        for split in source_splits.values()
    ]
    return max(src_steps)


def source_balanced_batches(
    source_splits: dict[str, DomainSplit],
    train_transform,
    source_per_domain: int = SOURCE_EXAMPLES_PER_DOMAIN_PER_BATCH,
    seed: int = SEED,
    epoch: int = 0,
):
    """
    Task 3 (Domain Generalization) batch generator: domain-balanced across
    the three SOURCE domains only (e.g. 8+8+8=24/step). This is the ONLY
    batching function task3/train.py is allowed to use -- unlike
    domain_balanced_batches above (Task 2/UDA), it never touches
    build_or_load_target_pool or any Sketch path, so the unseen-target
    protocol can't accidentally leak through the training loop.

    Same (seed, epoch)-determined per-domain cyclic shuffling as
    domain_balanced_batches, so it's reproducible the same way.
    """
    domain_names = list(source_splits.keys())
    epoch_seed = seed + epoch

    samplers = {
        domain: _CyclicIndexSampler(len(split.train_paths), seed=epoch_seed + i)
        for i, (domain, split) in enumerate(source_splits.items())
    }

    n_steps = steps_per_epoch_source_only(source_splits, source_per_domain)

    for _ in range(n_steps):
        source_images, source_labels, source_domain_ids = [], [], []
        for domain_id, domain in enumerate(domain_names):
            split = source_splits[domain]
            idxs = samplers[domain].next(source_per_domain)
            for idx in idxs:
                img = Image.open(split.train_paths[idx]).convert("RGB")
                source_images.append(train_transform(img))
                source_labels.append(split.train_labels[idx])
                source_domain_ids.append(domain_id)

        yield {
            "source_images": torch.stack(source_images),
            "source_labels": torch.tensor(source_labels, dtype=torch.long),
            "source_domain_ids": torch.tensor(source_domain_ids, dtype=torch.long),
            "domain_names": domain_names,  # index -> name, for logging/debugging
        }


def make_source_eval_loader(split: DomainSplit, eval_transform, batch_size: int = 64) -> DataLoader:
    """Non-shuffled loader over one source domain's val split, for checkpoint selection."""
    samples = [
        PACSSample(path=p, label=l, domain=split.domain)
        for p, l in zip(split.val_paths, split.val_labels)
    ]
    dataset = PACSDataset(samples, eval_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def make_target_eval_loader(
    target_samples: list[PACSSample], eval_transform, batch_size: int = 64
) -> DataLoader:
    """
    Non-shuffled loader over the full Sketch pool for final evaluation.
    Only evaluate_final.py should read the labels this yields.
    """
    dataset = PACSDataset(target_samples, eval_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)