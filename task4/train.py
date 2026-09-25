"""
Entry point: python -m task4.train --config task4/configs/vanilla.yaml
                                or  --config task4/configs/gcsc.yaml
Run from the repo root. Trains a ten-class CIFAR ResNet-18 (Vanilla or
GCSC -- identical recipe except GCSC's RandAugment, see task4/methods/gcsc.py)
with cross-entropy, SGD + cosine decay, selecting the checkpoint with the
highest CIFAR-10 VALIDATION ACCURACY -- per spec this is accuracy-based,
not macro-F1 like Tasks 2/3. Fixed 100-epoch budget, no early stopping
(the spec doesn't call for it here).

CIFAR-100 is never imported by this file -- only extract_outputs.py /
evaluate_osr.py touch it, and only after a checkpoint here is frozen.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from task4.data.cifar10 import build_datasets, get_gcsc_train_transform, get_transforms
from task4.methods import gcsc, proser, vanilla

METHOD_MODULES = {"vanilla": vanilla, "gcsc": gcsc, "proser": proser}


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults_name = cfg.pop("defaults", None)
    if defaults_name:
        with open(path.parent / defaults_name) as f:
            base_cfg = yaml.safe_load(f)
        return _deep_merge(base_cfg, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_method(cfg: dict):
    method = cfg["method"]
    if method not in METHOD_MODULES:
        raise NotImplementedError(f"Unknown Task 4 method '{method}'.")
    return METHOD_MODULES[method]


@torch.no_grad()
def evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval_mode()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.head(model.backbone(images))
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(cfg: dict) -> dict:
    set_all_seeds(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    method_module = build_method(cfg)

    eval_transform = get_transforms("eval")
    train_transform = get_gcsc_train_transform() if cfg["method"] == "gcsc" else get_transforms("train")

    train_dataset, val_dataset, test_dataset = build_datasets(
        cfg["data"]["cifar10_root"], train_transform, eval_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["optim"]["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = method_module.build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        momentum=cfg["optim"]["momentum"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["optim"]["max_epochs"])

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("run_name", cfg["method"])
    best_ckpt_path = checkpoint_dir / f"{run_name}_best.pt"

    history = []
    best_val_acc = -1.0

    for epoch in range(cfg["optim"]["max_epochs"]):
        model.train_mode()
        epoch_metrics: dict = {}
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            metrics = method_module.training_step(model, batch, criterion, device)
            optimizer.step()
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
            if step % 50 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={metrics['loss']:.4f} batch_acc={metrics['batch_acc']:.4f}"
                )
        scheduler.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        train_metric_means = {f"train_{k}": sum(v) / len(v) for k, v in epoch_metrics.items()}
        record = {
            "epoch": epoch,
            "val_accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0],
            **train_metric_means,
        }
        history.append(record)
        print(
            f"epoch {epoch}: train_loss={record['train_loss']:.4f} "
            f"val_accuracy={val_acc:.4f} lr={record['lr']:.5f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            method_module.save_checkpoint(model, str(best_ckpt_path))

    method_module.load_checkpoint(model, str(best_ckpt_path), device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\nFinal CIFAR-10 test accuracy ({run_name}): {test_acc:.4f}")

    results = {
        "history": history,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
    }
    with open(results_dir / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
