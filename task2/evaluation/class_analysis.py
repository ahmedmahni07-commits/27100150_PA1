"""
Task 2 "Common Evaluation" per-class target analysis. Target LABELS are used
ONLY here -- after every checkpoint, method, and threshold has already been
fixed -- never for training, model selection, or hyperparameter choice.
"""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from common.pacs_protocol import CLASSES

NUM_CLASSES = len(CLASSES)


@torch.no_grad()
def predict_all(model, loader, device):
    """Runs `model` on a labeled loader; returns (y_true, y_pred) as numpy arrays."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits, _ = model(images)
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def per_class_accuracy(y_true, y_pred, num_classes=NUM_CLASSES):
    """Accuracy restricted to each true class, length-`num_classes` array."""
    accs = np.zeros(num_classes)
    for c in range(num_classes):
        mask = y_true == c
        accs[c] = (y_pred[mask] == c).mean() if mask.sum() > 0 else np.nan
    return accs


def per_class_deltas(baseline_true, baseline_pred, method_true, method_pred,
                      num_classes=NUM_CLASSES):
    """Per-class accuracy change of `method` relative to `baseline` (e.g.
    Source-only) on the target set. Both should be evaluated on the full
    target test set (order need not match, only the label set)."""
    base_acc = per_class_accuracy(baseline_true, baseline_pred, num_classes)
    method_acc = per_class_accuracy(method_true, method_pred, num_classes)
    delta = method_acc - base_acc
    order = np.argsort(delta)  # most-degraded first ... most-improved last
    return {
        "per_class_accuracy_baseline": {CLASSES[c]: float(base_acc[c]) for c in range(num_classes)},
        "per_class_accuracy_method": {CLASSES[c]: float(method_acc[c]) for c in range(num_classes)},
        "delta": {CLASSES[c]: float(delta[c]) for c in range(num_classes)},
        "most_improved": [CLASSES[c] for c in order[::-1][:3]],
        "most_degraded": [CLASSES[c] for c in order[:3]],
    }


def dominant_confusions(y_true, y_pred, num_classes=NUM_CLASSES, top_k=3):
    """For each true class, the top_k most frequent *wrong* predicted classes."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    result = {}
    for c in range(num_classes):
        row = cm[c].copy()
        row[c] = 0  # exclude the correct-prediction count itself
        top = np.argsort(row)[::-1][:top_k]
        result[CLASSES[c]] = [(CLASSES[p], int(row[p])) for p in top if row[p] > 0]
    return result