"""
Usage: python task2/evaluation/plot_training_curve.py task2/results/<method>_results.json

Auto-detects whether the run has adversarial diagnostics (DANN/CDAN: history
entries carry train_domain_loss/train_domain_acc/train_grl_alpha) and
switches to a richer 3-panel diagnostic plot; otherwise falls back to the
simple 2-panel loss/macro-F1 plot (source_only/DAN).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _is_adversarial(history: list) -> bool:
    return bool(history) and "train_domain_loss" in history[0]


def plot_training_curve(results_path: str, out_path: str | None = None) -> None:
    with open(results_path) as f:
        results = json.load(f)
    history = results["history"]
    out_path = out_path or str(Path(results_path).with_suffix("")) + "_curve.png"

    if _is_adversarial(history):
        _plot_adversarial(history, out_path)
    else:
        _plot_simple(history, out_path)

    print(f"Saved plot to {out_path}")


def _plot_simple(history: list, out_path: str) -> None:
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    mean_val_f1 = [r["mean_val_macro_f1"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss (mean over batches)")
    ax1.set_title("Training loss")

    ax2.plot(epochs, mean_val_f1, marker="o", color="tab:green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean source-val macro-F1")
    ax2.set_title("Validation macro-F1 (mean over Photo/Art/Cartoon)")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _plot_adversarial(history: list, out_path: str) -> None:
    epochs = [r["epoch"] for r in history]
    cls_loss = [r["train_cls_loss"] for r in history]
    domain_loss = [r["train_domain_loss"] for r in history]
    domain_acc = [r["train_domain_acc"] for r in history]
    grl_alpha = [r["train_grl_alpha"] for r in history]
    mean_val_f1 = [r["mean_val_macro_f1"] for r in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))

    # Panel 1: classification loss vs domain loss, LOG scale -- these can
    # span many orders of magnitude when training is unstable, which is
    # exactly the story this panel needs to show.
    ax1.plot(epochs, cls_loss, marker="o", label="cls_loss", color="tab:blue")
    ax1.plot(epochs, domain_loss, marker="s", label="domain_loss", color="tab:red")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Classification vs. domain loss")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Panel 2: domain discriminator accuracy (chance = 0.5 reference line)
    # against the GRL's ramping alpha, on twin y-axes.
    ax2.plot(epochs, domain_acc, marker="o", color="tab:purple", label="domain_acc")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Domain discriminator accuracy", color="tab:purple")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax2.set_title("Domain confusion vs. GRL strength")

    ax2b = ax2.twinx()
    ax2b.plot(epochs, grl_alpha, marker="^", color="tab:orange", label="grl_alpha")
    ax2b.set_ylabel("GRL alpha", color="tab:orange")
    ax2b.set_ylim(0, 1)
    ax2b.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # Panel 3: the metric that actually matters -- source validation macro-F1.
    ax3.plot(epochs, mean_val_f1, marker="o", color="tab:green")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean source-val macro-F1")
    ax3.set_ylim(0, 1)
    ax3.set_title("Validation macro-F1 (mean over Photo/Art/Cartoon)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    plot_training_curve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
