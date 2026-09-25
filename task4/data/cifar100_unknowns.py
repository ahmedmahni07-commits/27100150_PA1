"""
CIFAR-100 unknown-class pools for Task 4's OSR evaluation. CIFAR-100 is
evaluation-only: none of its images (train OR test) may influence
training, checkpoint selection, score design, or threshold selection --
only the FINAL evaluation script (extract_outputs.py / evaluate_osr.py)
ever imports this module. The near/far groupings are FIXED for the whole
task and must not be revised after seeing results.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

# Fixed per spec -- do not revise after seeing results.
NEAR_UNKNOWN_CLASSES = [
    "bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel",
]
FAR_UNKNOWN_CLASSES = [
    "bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe",
]


class UnknownSubset(Dataset):
    """
    Wraps CIFAR-100's TEST split, filtered down to one fixed group's fine
    classes, with its own transform. Labels here are CIFAR-100 FINE-label
    indices, not CIFAR-10 known-class indices -- they are never fed to the
    classifier as targets, only used (if at all) for qualitative failure
    inspection (e.g. "this rejected/accepted image is class `wolf`").
    """

    def __init__(self, base_dataset, indices: list[int], transform, group_name: str):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.group_name = group_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, fine_label = self.base_dataset[self.indices[i]]
        fine_class_name = self.base_dataset.classes[fine_label]
        return self.transform(img), fine_label, fine_class_name


def build_unknown_datasets(data_root: str | Path, eval_transform):
    """
    Returns (near_dataset, far_dataset). Both draw only from CIFAR-100's
    TEST partition, filtered to their fixed 8 fine classes (100
    images/class x 8 classes = 800/group, matching the spec exactly --
    every available test image of those classes is used, no further
    subsampling needed).
    """
    base = CIFAR100(root=str(data_root), train=False, download=True, transform=None)
    class_to_idx = base.class_to_idx

    def _indices_for(class_names: list[str]) -> list[int]:
        missing = [n for n in class_names if n not in class_to_idx]
        if missing:
            raise KeyError(
                f"CIFAR-100 fine class name(s) not found (check spelling): {missing}"
            )
        target_fine_ids = {class_to_idx[name] for name in class_names}
        return [i for i, label in enumerate(base.targets) if label in target_fine_ids]

    near_indices = _indices_for(NEAR_UNKNOWN_CLASSES)
    far_indices = _indices_for(FAR_UNKNOWN_CLASSES)

    near_dataset = UnknownSubset(base, near_indices, eval_transform, group_name="near")
    far_dataset = UnknownSubset(base, far_indices, eval_transform, group_name="far")
    return near_dataset, far_dataset
