"""
Entry point: python -m task4.extract_outputs --method vanilla
                                   (or --method gcsc / --method proser)

Freezes the selected checkpoint for `method` and runs it ONCE over every
example set Task 4 needs downstream (CIFAR-10 train/val/test + the fixed
CIFAR-100 near/far unknown pools), saving penultimate features and logits
to task4/cache/<run_name>_outputs.npz. Every score (MSP/MLS/Energy/
Mahalanobis) and evaluate_osr.py reads from this cache instead of
re-running the network, so every score is guaranteed to use exactly the
same saved logits/features (spec's requirement).

CIFAR-10's TRAIN split here uses the EVAL transform (no crop/flip/
RandAugment) -- "unaugmented CIFAR-10 training features", exactly what
the Mahalanobis fit requires -- a fresh unaugmented pass, independent of
whatever transform the checkpoint was actually trained with.

For PROSER, also saves dummy_logits (the num_dummy_classifiers raw
outputs) alongside known_logits, so evaluate_osr.py can compute both the
ten-known-class MLS score AND PROSER's own placeholder-based detection
score from one cache file. CIFAR-100 near/far entries additionally save
their CIFAR-100 fine-class name per example (for qualitative failure
inspection), since UnknownSubset yields a 3rd element ordinary CIFAR-10
loaders don't.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from task4.data.cifar10 import build_datasets, get_transforms
from task4.data.cifar100_unknowns import build_unknown_datasets
from task4.methods import gcsc, proser, vanilla
from task4.train import get_device, load_config

METHOD_MODULES = {"vanilla": vanilla, "gcsc": gcsc, "proser": proser}
METHOD_CONFIGS = {
    "vanilla": "task4/configs/vanilla.yaml",
    "gcsc": "task4/configs/gcsc.yaml",
    "proser": "task4/configs/proser.yaml",
}


@torch.no_grad()
def _run_over_loader(model, loader: DataLoader, device: torch.device, has_dummy_head: bool) -> dict:
    model.eval_mode()
    all_features, all_known_logits, all_dummy_logits, all_labels, all_class_names = [], [], [], [], []
    has_class_names = False

    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        features = model.backbone(images)
        known_logits = model.head(features)

        all_features.append(features.cpu().numpy())
        all_known_logits.append(known_logits.cpu().numpy())
        all_labels.append(np.asarray(labels))
        if has_dummy_head:
            all_dummy_logits.append(model.dummy_head(features).cpu().numpy())
        if len(batch) > 2:
            has_class_names = True
            all_class_names.extend(batch[2])

    out = {
        "features": np.concatenate(all_features),
        "known_logits": np.concatenate(all_known_logits),
        "labels": np.concatenate(all_labels),
    }
    if has_dummy_head:
        out["dummy_logits"] = np.concatenate(all_dummy_logits)
    if has_class_names:
        out["class_names"] = np.array(all_class_names, dtype=object)
    return out


def extract_for_method(method: str, device: torch.device) -> dict:
    cfg = load_config(METHOD_CONFIGS[method])
    method_module = METHOD_MODULES[method]
    has_dummy_head = method == "proser"

    model = method_module.build_model(cfg, device)
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    run_name = cfg.get("run_name", cfg["method"])
    ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} -- run "
            f"`python -m task4.train --config {METHOD_CONFIGS[method]}` first."
        )
    method_module.load_checkpoint(model, str(ckpt_path), device)

    eval_transform = get_transforms("eval")
    train_dataset, val_dataset, test_dataset = build_datasets(
        cfg["data"]["cifar10_root"], eval_transform, eval_transform  # unaugmented train pass too
    )
    near_dataset, far_dataset = build_unknown_datasets(cfg["data"]["cifar100_root"], eval_transform)

    loaders = {
        "cifar10_train": DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=2),
        "cifar10_val": DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2),
        "cifar10_test": DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2),
        "cifar100_near": DataLoader(near_dataset, batch_size=256, shuffle=False, num_workers=2),
        "cifar100_far": DataLoader(far_dataset, batch_size=256, shuffle=False, num_workers=2),
    }

    outputs = {}
    for split_name, loader in loaders.items():
        print(f"  extracting {split_name} ({len(loader.dataset)} examples)...")
        outputs[split_name] = _run_over_loader(model, loader, device, has_dummy_head)

    cache_dir = Path(cfg["logging"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{run_name}_outputs.npz"

    flat = {}
    for split_name, split_out in outputs.items():
        for key, arr in split_out.items():
            flat[f"{split_name}__{key}"] = arr
    np.savez_compressed(out_path, **flat)
    print(f"Saved cached outputs to {out_path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHOD_MODULES.keys()))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"=== Extracting outputs for {args.method} ===")
    extract_for_method(args.method, device)


if __name__ == "__main__":
    main()
