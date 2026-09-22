"""
Task 2 training entry point. Config-driven: the same loop trains any of
source_only / dan / dann / cdan, since every method class implements the
shared compute_loss(src_images, src_labels, tgt_images, progress) API.

Usage (run from the atml-pa1 repo root, inside your venv):
    python -m task2.train --config task2/configs/source_only.yaml
    python -m task2.train --config task2/configs/dan.yaml
    python -m task2.train --config task2/configs/dann.yaml
    python -m task2.train --config task2/configs/cdan.yaml
"""
import argparse
import os

import torch
import yaml

from common.pacs import MultiDomainBatchStream, make_loader, steps_per_epoch_for
from common.pacs_protocol import SOURCE_DOMAINS, build_source_splits, target_pool
from common.plotting import plot_training_curves
from common.seed import set_seed
from task2.evaluation.metrics import evaluate_source_domains
from task2.models.backbone import freeze_bn_running_stats


def build_model(method_name, num_classes=7, lambda_mmd=1.0, lambda_dom=1.0):
    if method_name == "source_only":
        from task2.methods.source_only import SourceOnly
        return SourceOnly(num_classes=num_classes)
    if method_name == "dan":
        from task2.methods.dan import DAN
        return DAN(num_classes=num_classes, lambda_mmd=lambda_mmd)
    if method_name == "dann":
        from task2.methods.dann import DANN
        return DANN(num_classes=num_classes, lambda_dom=lambda_dom)
    if method_name == "cdan":
        from task2.methods.cdan import CDAN
        return CDAN(num_classes=num_classes, lambda_dom=lambda_dom)
    raise ValueError(f"Unknown method: {method_name!r}")


def load_config(path):
    """Minimal hand-rolled stand-in for Hydra's `defaults: [base]` list --
    every task2/configs/*.yaml only ever references base.yaml, so this just
    merges that one file in when `defaults` is present."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if "defaults" in cfg and os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        return {**base, **{k: v for k, v in cfg.items() if k != "defaults"}}
    return cfg


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(cfg, needs_target, num_workers=4):
    splits = build_source_splits()
    train_loaders, val_loaders = {}, {}
    for domain in SOURCE_DOMAINS:
        train_loaders[domain] = make_loader(
            splits[domain]["train"], split="train",
            batch_size=cfg["batch_size_source"], shuffle=True,
            return_label=True, num_workers=num_workers, drop_last=True,
        )
        val_loaders[domain] = make_loader(
            splits[domain]["val"], split="eval",
            batch_size=64, shuffle=False,
            return_label=True, num_workers=num_workers, drop_last=False,
        )

    target_loader = None
    if needs_target:
        # return_label=False: the loader physically cannot hand a target
        # label to the training loop, even though the underlying split has
        # them (see common/pacs.py) -- this is the target-leakage guard.
        target_loader = make_loader(
            target_pool(), split="train",
            batch_size=cfg["batch_size_target"], shuffle=True,
            return_label=False, num_workers=num_workers, drop_last=True,
        )
    return train_loaders, val_loaders, target_loader


def train(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 6304))

    method_name = cfg["method"]
    needs_target = method_name != "source_only"

    device = get_device()
    train_loaders, val_loaders, target_loader = build_dataloaders(cfg, needs_target)

    steps_per_epoch = steps_per_epoch_for(train_loaders)
    batch_stream = MultiDomainBatchStream(train_loaders, steps_per_epoch, target_loader)

    model = build_model(
        method_name, num_classes=7,
        lambda_mmd=cfg.get("lambda_mmd", 1.0),
        lambda_dom=cfg.get("lambda_dom", 1.0),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    max_epochs = cfg["max_epochs"]
    patience = cfg["patience"]
    # DANN/CDAN's alpha(p) schedule is normalized against the *planned* total
    # steps, not however many are actually run before early stopping --
    # that's a standard approximation, not a bug.
    total_steps = max(1, max_epochs * steps_per_epoch)

    best_mean_f1 = -1.0
    epochs_without_improvement = 0
    history = {"loss": [], "mean_f1": [], "domains": {d: [] for d in SOURCE_DOMAINS}}

    os.makedirs("task2/results", exist_ok=True)
    ckpt_path = f"task2/results/best_{method_name}.pth"

    global_step = 0
    print(f"[{method_name}] device={device}  steps/epoch={steps_per_epoch}  "
          f"planned_total_steps={total_steps}")

    for epoch in range(max_epochs):
        model.train()
        model.apply(freeze_bn_running_stats)  # .train() re-enables BN stat updates; undo it every epoch

        epoch_loss, n_steps = 0.0, 0
        for source_batch, target_batch in batch_stream:
            progress = global_step / (total_steps - 1) if total_steps > 1 else 0.0
            src_images = torch.cat([b[0] for b in source_batch.values()], dim=0).to(device)
            src_labels = torch.cat([b[1] for b in source_batch.values()], dim=0).to(device)
            tgt_images = target_batch.to(device) if target_batch is not None else None

            optimizer.zero_grad()
            loss, _ = model.compute_loss(src_images, src_labels, tgt_images, progress=progress)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1
            global_step += 1

        avg_loss = epoch_loss / n_steps
        per_domain_f1, mean_f1 = evaluate_source_domains(model, val_loaders, device)

        history["loss"].append(avg_loss)
        history["mean_f1"].append(mean_f1)
        for d, f1 in per_domain_f1.items():
            history["domains"][d].append(f1)

        print(f"epoch {epoch + 1}/{max_epochs}  loss={avg_loss:.4f}  mean_val_f1={mean_f1:.4f}")
        for d, f1 in per_domain_f1.items():
            print(f"    {d:>14}: f1={f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> new best ({best_mean_f1:.4f}), saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"    -> no improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    os.makedirs("report/figures", exist_ok=True)
    plot_training_curves(history, save_path=f"report/figures/{method_name}_learning_curve.png")
    print(f"done. best mean source-val macro-F1 = {best_mean_f1:.4f}  checkpoint = {ckpt_path}")
    return ckpt_path, best_mean_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
