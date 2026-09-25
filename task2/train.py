"""
Entry point: python -m task2.train --config task2/configs/source_only.yaml
Run from the repo root (atml-pa1/) so the `common.` / `task2.` imports and
the relative results/checkpoint paths in the config resolve correctly.
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
    build_or_load_target_pool,
    domain_balanced_batches,
    make_source_eval_loader,
    make_target_eval_loader,
    set_all_seeds,
    steps_per_epoch,
)
from task2.evaluation.metrics import evaluate_loader
from task2.methods import source_only, dan, dann, cdan


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
    if method == "source_only":
        return source_only
    if method == "dan":
        return dan
    if method == "dann":
        return dann
    if method == "cdan":
        return cdan
    raise NotImplementedError(f"Unknown method '{method}'.")


def train(cfg: dict) -> dict:
    set_all_seeds(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    method_module = build_method(cfg)

    source_splits = build_or_load_source_splits(cfg["data"]["pacs_root"])
    target_samples = build_or_load_target_pool(cfg["data"]["pacs_root"])

    train_transform = get_backbone_transforms("train", cfg["data"]["resize"], cfg["data"]["crop"])
    eval_transform = get_backbone_transforms("val", cfg["data"]["resize"], cfg["data"]["crop"])

    val_loaders = {d: make_source_eval_loader(s, eval_transform) for d, s in source_splits.items()}
    target_eval_loader = make_target_eval_loader(target_samples, eval_transform)

    model = method_module.build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"]
    )

    n_steps = steps_per_epoch(
        source_splits, target_samples, cfg["batch"]["source_per_domain"], cfg["batch"]["target_per_batch"]
    )
    print(f"{n_steps} steps/epoch")

    history = []
    best_mean_macro_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # run_name lets a config override the checkpoint/results filename so
    # e.g. a controlled-study sweep over dann.yaml variants doesn't clobber
    # the main comparison's dann_best.pt / dann_results.json. Defaults to
    # the method name, matching every existing config's behavior.
    run_name = cfg.get("run_name", cfg["method"])
    best_ckpt_path = checkpoint_dir / f"{run_name}_best.pt"

    # progress_p (used by DANN/CDAN's gradient-reversal schedule) is defined
    # relative to the FULL planned training budget (max_epochs * n_steps),
    # not the actual run length -- early stopping cutting a run short just
    # means alpha never reaches its max, which is fine and consistent
    # across methods since max_epochs/n_steps are identical for all of them.
    total_steps = cfg["optim"]["max_epochs"] * n_steps
    global_step = 0

    for epoch in range(cfg["optim"]["max_epochs"]):
        model.train_mode()
        epoch_metrics: dict = {}  # key -> list of per-step values, whatever training_step returns
        batch_iter = domain_balanced_batches(
            source_splits, target_samples, train_transform,
            source_per_domain=cfg["batch"]["source_per_domain"],
            target_per_batch=cfg["batch"]["target_per_batch"],
            seed=cfg["seed"], epoch=epoch,
        )
        for step, batch in enumerate(batch_iter):
            progress_p = global_step / total_steps if total_steps > 0 else 0.0
            optimizer.zero_grad()
            metrics = method_module.training_step(model, batch, criterion, device, progress_p=progress_p)
            # DEVIATION from manual (see task2/DEVIATIONS.md): optional gradient-norm
            # clipping, applied after backward() (inside training_step) and before the
            # optimizer step. A no-op whenever cfg["optim"]["grad_clip_norm"] is null/absent
            # -- i.e. exactly the source_only/DAN configs, which are unchanged.
            grad_clip_norm = cfg["optim"].get("grad_clip_norm")
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                metrics["grad_norm"] = grad_norm.item()
            optimizer.step()
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
            global_step += 1
            if step % 20 == 0:
                extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "loss")
                print(f"  epoch {epoch} step {step}/{n_steps} loss={metrics['loss']:.4f} p={progress_p:.3f} {extra}")

        per_domain_val = {}
        for domain, loader in val_loaders.items():
            result = evaluate_loader(model, loader, device)
            per_domain_val[domain] = {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
        mean_macro_f1 = sum(v["macro_f1"] for v in per_domain_val.values()) / len(per_domain_val)

        # Every key training_step returned (loss, cls_loss, mmd_loss/domain_loss,
        # batch_acc, domain_acc, grl_alpha, ...) gets averaged over the epoch and
        # logged as train_<key> -- this is the "classification and alignment or
        # domain-loss curves" evidence Step 5 asks for, generic across all methods.
        train_metric_means = {f"train_{k}": sum(v) / len(v) for k, v in epoch_metrics.items()}

        record = {
            "epoch": epoch,
            "train_loss": train_metric_means["train_loss"],
            "mean_val_macro_f1": mean_macro_f1,
            **{k: v for k, v in train_metric_means.items() if k != "train_loss"},
            **{f"{d}_val_acc": v["accuracy"] for d, v in per_domain_val.items()},
            **{f"{d}_val_macro_f1": v["macro_f1"] for d, v in per_domain_val.items()},
        }
        history.append(record)
        extra_str = " ".join(f"{k}={v:.4f}" for k, v in train_metric_means.items() if k != "train_loss")
        print(f"epoch {epoch}: train_loss={record['train_loss']:.4f} mean_val_macro_f1={mean_macro_f1:.4f} {extra_str}")

        if mean_macro_f1 > best_mean_macro_f1:
            best_mean_macro_f1 = mean_macro_f1
            epochs_without_improvement = 0
            method_module.save_checkpoint(model, str(best_ckpt_path))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["optim"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improvement} epochs)")
                break

    method_module.load_checkpoint(model, str(best_ckpt_path), device)

    final_per_domain_val = {}
    for domain, loader in val_loaders.items():
        result = evaluate_loader(model, loader, device)
        final_per_domain_val[domain] = {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}

    results = {
        "history": history,
        "best_mean_val_macro_f1": best_mean_macro_f1,
        "final_per_domain_val": final_per_domain_val,
    }

    # Source-only's own target performance is required to be reported at
    # THIS stage ("report its final performance on the target" — step 1).
    # DAN/DANN/CDAN target numbers are gathered together LATER, in
    # evaluate_final.py, once every method's config is frozen — don't do
    # this early-peek for those methods.
    if cfg["method"] == "source_only":
        target_result = evaluate_loader(model, target_eval_loader, device)
        results["target"] = {"accuracy": target_result["accuracy"], "macro_f1": target_result["macro_f1"]}

    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()