"""
PACS Dataset, transforms, and multi-domain batch construction shared by
Tasks 2 and 3.

This module only depends on common/pacs_protocol.py for *which* images go
where; it owns *how* those images are loaded and batched.
"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common.pacs_protocol import IMAGES_ROOT

# Standard ImageNet normalization -- required since every backbone (ResNet-18
# here, ResNet-50/ViT/CLIP in Task 1) was pretrained on ImageNet-normalized inputs.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(split):
    """split='train': Resize 256 -> RandomCrop 224 -> RandomHorizontalFlip
                       (matches the assignment's augmentation spec exactly).
       split='eval' : Resize 256 -> CenterCrop 224, no randomness -- used for
                       both source validation and final target evaluation."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if split == "eval":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(f"Unknown split: {split!r}")


class PACSDataset(Dataset):
    """Wraps a list of (relpath, label) samples for one PACS domain.

    return_label=False is used for the target domain during adaptation: the
    images are still loaded, but no label ever leaves this object, so a
    downstream training loop cannot accidentally consume target labels.
    """

    def __init__(self, samples, transform, return_label=True):
        self.samples = samples
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        relpath, label = self.samples[idx]
        img = Image.open(os.path.join(IMAGES_ROOT, relpath)).convert("RGB")
        img = self.transform(img)
        if self.return_label:
            return img, label
        return img


def make_loader(samples, split, batch_size, shuffle, return_label=True,
                 num_workers=4, drop_last=False):
    ds = PACSDataset(samples, build_transform(split), return_label=return_label)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def _infinite(loader):
    """Re-iterate a DataLoader forever without caching yielded batches in memory
    (unlike itertools.cycle, which would hold every past image tensor).
    Each pass reshuffles if the loader was built with shuffle=True."""
    while True:
        for batch in loader:
            yield batch


class MultiDomainBatchStream:
    """One balanced mini-batch per step: a batch from each source-domain loader
    (cycling shorter domains so every step is fully populated), optionally
    paired with a batch from an unlabeled target loader.

    Implements the assignment's "8 examples from each source domain and 24
    target examples ... cycle a loader when necessary" requirement, and is
    reused as-is by Task 3 with target_loader=None.
    """

    def __init__(self, source_loaders: dict, steps_per_epoch: int, target_loader=None):
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        source_iters = {d: _infinite(loader) for d, loader in self.source_loaders.items()}
        target_iter = _infinite(self.target_loader) if self.target_loader is not None else None
        for _ in range(self.steps_per_epoch):
            source_batch = {d: next(it) for d, it in source_iters.items()}
            target_batch = next(target_iter) if target_iter is not None else None
            yield source_batch, target_batch


def steps_per_epoch_for(loaders: dict):
    """One epoch = enough steps to see the *largest* domain once; smaller
    domains are cycled (see MultiDomainBatchStream / _infinite)."""
    return max(len(loader) for loader in loaders.values())
