"""
Entry point: python -m task3.train --config task3/configs/dan_dg.yaml
Run from the repo root (atml-pa1/) so the `common.` / `task2.` / `task3.`
imports and the relative results/checkpoint paths in the config resolve
correctly.

CRITICAL protocol constraint (Domain Generalization, not UDA): Sketch is
NEVER loaded here. This module only ever imports
common.pacs_protocol.build_or_load_source_splits /
common.pacs_protocol.source_balanced_batches / .steps_per_epoch_source_only
-- it never imports build_or_load_target_pool, make_target_eval_loader, or
anything else that would expose the target domain. Sketch is loaded for
the first and only time in task3/evaluate_sketch.py, after every
checkpoint here is already frozen.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from common.pacs import get_backbone_transforms
from common.pacs_protocol import (
    build_or_load_source_splits,
    make_source_eval_loader,
    set_all_seeds,
    source_balanced_batches,
    steps_per_epoch_source_only,
)
from task3.methods import erm, dan_dg, sam
from task3.selection.source_validation import (
    SourceValidationSelector,
    evaluate_all_source_domains,
    mean_source_macro_f1,
)


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


def build_method(cfg: dict):
    method = cfg["method"]
    if method == "erm":
        return erm
    if method == "dan_dg":
        return dan_dg
    if method == "sam":
        return sam
    raise NotImplementedError(f"Unknown Task 3 method '{method}'.")


def _train_erm(cfg, method_module, device, val_loaders, checkpoint_dir, results_dir, run_name) -> dict:
    """
    ERM is Task 2's frozen source-only checkpoint, reused unchanged -- no
    training loop at all. Just: load it, evaluate it on source-val (for
    the record), and save a copy under task3's own checkpoint dir so
    evaluate_sketch.py can load every method (erm / dan_dg / sam) the same
    uniform way later.
    """
    model = method_module.build_model(cfg, device)
    model = method_module.load_source_checkpoint(model, cfg, device)

    per_domain_val = evaluate_all_source_domains(model, val_loaders, device)
    mean_f1 = mean_source_macro_f1(per_domain_val)
    print(f"[erm] reused Task 2 checkpoint -- mean source-val macro-F1={mean_f1:.4f}")

    best_ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
    method_module.save_checkpoint(model, str(best_ckpt_path))

    results = {
        "note": "ERM baseline is Task 2's source-only checkpoint, reused unchanged (not retrained in Task 3).",
        "source_checkpoint": cfg["erm"]["source_checkpoint"],
        "final_per_domain_val": per_domain_val,
        "mean_val_macro_f1": mean_f1,
    }
    with open(results_dir / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


def _run_epoch_loop(
    cfg, method_module, device, model, source_splits, train_transform, val_loaders,
    checkpoint_dir, results_dir, run_name, step_fn,
) -> dict:
    """
    Shared epoch/logging/selection/checkpointing scaffold for dan_dg and
    sam. `step_fn(model, batch, criterion, device)` runs exactly ONE
    training step (however many forward/backward passes it needs
    internally) and returns the metrics dict to log for that step -- this
    is where dan_dg's single-pass training_step and sam's two-pass
    ascent/descent dance are plugged in without duplicating everything
    else (history bookkeeping, early stopping, checkpoint I/O).
    """
    criterion = nn.CrossEntropyLoss()
    n_steps = steps_per_epoch_source_only(source_splits, cfg["batch"]["source_per_domain"])
    print(f"{n_steps} steps/epoch")

    selector = SourceValidationSelector(patience=cfg["optim"]["early_stopping_patience"])
    best_ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
    history = []

    for epoch in range(cfg["optim"]["max_epochs"]):
        model.train_mode()
        epoch_metrics: dict = {}
        batch_iter = source_balanced_batches(
            source_splits, train_transform,
            source_per_domain=cfg["batch"]["source_per_domain"],
            seed=cfg["seed"], epoch=epoch,
        )
        for step, batch in enumerate(batch_iter):
            metrics = step_fn(model, batch, criterion, device)
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
            if step % 20 == 0:
                extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "loss")
                print(f"  epoch {epoch} step {step}/{n_steps} loss={metrics.get('loss', float('nan')):.4f} {extra}")

        per_domain_val = evaluate_all_source_domains(model, val_loaders, device)
        mean_f1 = mean_source_macro_f1(per_domain_val)
        train_metric_means = {f"train_{k}": sum(v) / len(v) for k, v in epoch_metrics.items()}

        record = {
            "epoch": epoch,
            "mean_val_macro_f1": mean_f1,
            **train_metric_means,
            **{f"{d}_val_acc": v["accuracy"] for d, v in per_domain_val.items()},
            **{f"{d}_val_macro_f1": v["macro_f1"] for d, v in per_domain_val.items()},
        }
        history.append(record)
        extra_str = " ".join(f"{k}={v:.4f}" for k, v in train_metric_means.items())
        print(f"epoch {epoch}: mean_val_macro_f1={mean_f1:.4f} {extra_str}")

        if selector.update(mean_f1):
            method_module.save_checkpoint(model, str(best_ckpt_path))
        else:
            if selector.should_stop():
                print(f"Early stopping at epoch {epoch} (no improvement for {selector.epochs_without_improvement} epochs)")
                break

    method_module.load_checkpoint(model, str(best_ckpt_path), device)
    final_per_domain_val = evaluate_all_source_domains(model, val_loaders, device)

    results = {
        "history": history,
        "best_mean_val_macro_f1": selector.best_score,
        "final_per_domain_val": final_per_domain_val,
    }
    with open(results_dir / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


def train(cfg: dict) -> dict:
    set_all_seeds(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    method_module = build_method(cfg)

    # Source splits only -- build_or_load_target_pool is never called
    # anywhere in this file, by design (Task 3 = Sketch stays unseen).
    source_splits = build_or_load_source_splits(cfg["data"]["pacs_root"])

    train_transform = get_backbone_transforms("train", cfg["data"]["resize"], cfg["data"]["crop"])
    eval_transform = get_backbone_transforms("val", cfg["data"]["resize"], cfg["data"]["crop"])
    val_loaders = {d: make_source_eval_loader(s, eval_transform) for d, s in source_splits.items()}

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("run_name", cfg["method"])

    if cfg["method"] == "erm":
        return _train_erm(cfg, method_module, device, val_loaders, checkpoint_dir, results_dir, run_name)

    model = method_module.build_model(cfg, device)

    if cfg["method"] == "sam":
        optimizer = sam.SAMOptimizer(
            model.parameters(), base_optimizer_cls=torch.optim.AdamW,
            rho=cfg["sam"]["rho"], lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"],
        )

        def step_fn(model, batch, criterion, device):
            optimizer.zero_grad()
            m1 = sam.training_step_first_pass(model, batch, criterion, device)
            optimizer.ascent_step()
            optimizer.zero_grad()
            m2 = sam.training_step_second_pass(model, batch, criterion, device)
            optimizer.descent_step()
            return {
                "loss": m2["loss"],
                "cls_loss": m2["cls_loss"],
                "batch_acc": m1["batch_acc"],
                "sharpness_gap": m2["loss"] - m1["first_pass_loss"],
            }

        return _run_epoch_loop(
            cfg, method_module, device, model, source_splits, train_transform, val_loaders,
            checkpoint_dir, results_dir, run_name, step_fn,
        )

    # dan_dg (and any future plain single-pass method)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"]
    )

    def step_fn(model, batch, criterion, device):
        optimizer.zero_grad()
        metrics = method_module.training_step(model, batch, criterion, device, progress_p=0.0)
        optimizer.step()
        return metrics

    return _run_epoch_loop(
        cfg, method_module, device, model, source_splits, train_transform, val_loaders,
        checkpoint_dir, results_dir, run_name, step_fn,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
