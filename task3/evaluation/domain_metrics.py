"""
Task 3 evaluation helpers: per-class Sketch accuracy and most-improved /
most-degraded-vs-ERM analysis, plus a flat comparison table builder across
methods (ERM / DAN-DG / SAM) for the final Required Evidence writeup.

Target/Sketch labels are used ONLY by evaluate_sketch.py at the very end --
this module is just pure functions over whatever y_true/y_pred arrays it's
handed, so importing it does not by itself violate the "Sketch never used
during training or selection" rule.

Reuses task2.evaluation.class_analysis's per-class helpers directly rather
than reimplementing them -- they're generic over any y_true/y_pred arrays
and carry no Task-2-specific assumptions.
"""

from __future__ import annotations

import numpy as np

from task2.evaluation.class_analysis import (
    per_class_accuracy,
    compare_to_baseline,
    confusion_matrix_for_class,
)

__all__ = [
    "per_class_accuracy",
    "compare_to_baseline",
    "confusion_matrix_for_class",
    "build_comparison_table",
]


def build_comparison_table(results_by_method: dict) -> dict:
    """
    results_by_method: {method_name: {"sketch": {"accuracy": ..., "macro_f1": ...}, ...}}
    (as produced by evaluate_sketch.py for each of erm/dan_dg/sam).

    Returns {method_name: {"sketch_accuracy", "sketch_macro_f1",
    "delta_accuracy_vs_erm", "delta_macro_f1_vs_erm"}} -- the flat table
    for "compare ERM vs. DAN-DG vs. SAM on the never-seen Sketch domain".
    """
    table = {}
    erm_sketch = results_by_method.get("erm", {}).get("sketch", {})
    erm_acc = erm_sketch.get("accuracy")
    erm_f1 = erm_sketch.get("macro_f1")

    for method, res in results_by_method.items():
        sketch = res.get("sketch", {})
        entry = {
            "sketch_accuracy": sketch.get("accuracy"),
            "sketch_macro_f1": sketch.get("macro_f1"),
        }
        if erm_acc is not None and sketch.get("accuracy") is not None:
            entry["delta_accuracy_vs_erm"] = sketch["accuracy"] - erm_acc
        if erm_f1 is not None and sketch.get("macro_f1") is not None:
            entry["delta_macro_f1_vs_erm"] = sketch["macro_f1"] - erm_f1
        table[method] = entry

    return table
