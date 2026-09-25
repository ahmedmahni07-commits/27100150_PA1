"""
Full 7x7 confusion matrices for the Common Evaluation and Alignment
Diagnostic (Step 5): one per method (Source-only, DAN, DANN, CDAN), built
from the target_y_true/target_y_pred saved by evaluate_final.py. Saves a
combined heatmap figure plus per-method CSVs (raw counts) so the numbers
are traceable to a file, not just a picture.

Usage: python3 task2/evaluation/plot_confusion_matrices.py
(needs only numpy + matplotlib -- run with plain python3 if your venv's
torch isn't needed for this step)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = ["source_only", "dan", "dann", "cdan"]
METHOD_LABELS = {"source_only": "Source-only", "dan": "DAN", "dann": "DANN", "cdan": "CDAN"}


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="task2/results/evaluate_final_results.json",
        help="Path to an evaluate_final(-style) results JSON containing target_y_true/target_y_pred per method.",
    )
    parser.add_argument(
        "--out-dir", default="task2/results/figures",
        help="Directory to write the CSVs and combined figure to.",
    )
    args = parser.parse_args()
    results_path = Path(args.results)
    fig_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text())
    num_classes = len(results["source_only"]["per_class_target_accuracy"])
    class_labels = [f"class {i}" for i in range(num_classes)]

    matrices = {}
    for method in METHOD_ORDER:
        y_true = results[method]["target_y_true"]
        y_pred = results[method]["target_y_pred"]
        cm = confusion_matrix(y_true, y_pred, num_classes)
        matrices[method] = cm

        # raw counts, saved as CSV -- traceable source for the report table
        csv_path = fig_dir / f"{method}_confusion_matrix.csv"
        with open(csv_path, "w") as f:
            f.write("true\\pred," + ",".join(class_labels) + "\n")
            for i, row in enumerate(cm):
                f.write(f"{class_labels[i]}," + ",".join(str(v) for v in row) + "\n")
        print(f"wrote {csv_path}")

    # combined 2x2 heatmap figure, row-normalized (recall per true class) so
    # methods with different target-set compositions are still comparable
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, method in zip(axes.flat, METHOD_ORDER):
        cm = matrices[method]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{METHOD_LABELS[method]} -- target (Sketch) confusion matrix")
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(range(num_classes))
        ax.set_yticklabels(range(num_classes))

        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j]
                if count == 0:
                    continue
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized (recall)")

    fig.suptitle("Task 2: target (Sketch) confusion matrices, all four methods\n(cell text = raw count, color = fraction of that true class)", y=1.01)
    fig.tight_layout()
    out_path = fig_dir / "confusion_matrices_all_methods.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
