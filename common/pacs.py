"""
PACS dataset loading utilities, shared by Task 2 (UDA) and Task 3 (DG).

PACS: 7 object classes across 4 domains (Photo, Art Painting, Cartoon, Sketch).
Sketch is always the held-out/target domain for Tasks 2 and 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

PACS_CLASSES = [
    "dog", "elephant", "giraffe", "guitar", "horse", "house", "person",
]
PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"

CLASS_TO_IDX = {c: i for i, c in enumerate(PACS_CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class PACSSample:
    path: str
    label: int
    domain: str


def list_domain_images(root: str | Path, domain: str) -> list[PACSSample]:
    """
    Expects the standard PACS on-disk layout:
        <root>/<domain>/<class_name>/*.jpg

    with domain folder names exactly matching PACS_DOMAINS and class folder
    names exactly matching PACS_CLASSES. If your copy of PACS uses different
    folder names (e.g. "art_painting" vs "art painting"), fix that at the
    filesystem level rather than here, so the cached split JSONs stay valid.
    """
    domain_dir = Path(root) / domain
    if not domain_dir.is_dir():
        raise FileNotFoundError(
            f"PACS domain directory not found: {domain_dir}. "
            f"Expected layout <root>/<domain>/<class_name>/*.jpg"
        )

    samples: list[PACSSample] = []
    for class_name in PACS_CLASSES:
        class_dir = domain_dir / class_name
        if not class_dir.is_dir():
            continue
        label = CLASS_TO_IDX[class_name]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(PACSSample(path=str(img_path), label=label, domain=domain))

    if not samples:
        raise RuntimeError(
            f"No images found under {domain_dir}. Check that class subfolder "
            f"names match {PACS_CLASSES}."
        )
    return samples


def get_backbone_transforms(
    split: str,
    resize: int = 256,
    crop: int = 224,
    normalize_mean: Optional[tuple[float, float, float]] = None,
    normalize_std: Optional[tuple[float, float, float]] = None,
) -> Callable:
    """
    train: resize 256x256 -> random 224x224 crop -> horizontal flip
    val/eval/test: resize 256x256 -> center 224x224 crop

    Normalization defaults to ResNet18_Weights.IMAGENET1K_V1's own mean/std
    (pulled from the weights' transforms) unless explicitly overridden —
    per spec: "Apply the normalization associated with the pretrained weights."
    """
    if normalize_mean is None or normalize_std is None:
        weight_tf = ResNet18_Weights.IMAGENET1K_V1.transforms()
        normalize_mean = weight_tf.mean
        normalize_std = weight_tf.std

    if split == "train":
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    if split in ("val", "eval", "test"):
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    raise ValueError(f"Unknown split: {split!r} (expected 'train' or 'val'/'eval'/'test')")


class PACSDataset(Dataset):
    """
    A flat dataset over one or more PACS domains.

    Returns (image_tensor, label, domain_name). `label` is loaded for every
    sample (including target/Sketch) because it exists on disk, but Task 2's
    method training loops (dan.py/dann.py/cdan.py) must never read it for
    target examples — only evaluate_final.py is allowed to.
    """

    def __init__(self, samples: list[PACSSample], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.transform(image)
        return image, sample.label, sample.domain