"""Shared accuracy / macro-F1 computation, used by all methods + evaluate_final.py."""

import numpy as np
import torch
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
    )


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    """
    Run `model` (exposing .backbone, .head, .eval_mode()) over `loader`.
    Returns accuracy, macro_f1, and the raw y_true/y_pred arrays so callers
    (e.g. class_analysis.py later) can reuse them without a second forward pass.
    """
    model.eval_mode()
    all_true, all_pred = [], []
    for images, labels, _domains in loader:
        images = images.to(device, non_blocking=True)
        features = model.backbone(images)
        logits = model.head(features)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(np.asarray(labels))
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }