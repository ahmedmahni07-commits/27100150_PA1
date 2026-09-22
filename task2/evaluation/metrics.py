"""
Accuracy / macro-F1 helpers shared across Task 2 methods -- used both for
source validation (checkpoint selection) and final target (Sketch) evaluation.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate_domain(model, loader, device):
    """Runs `model` (switched to eval mode) over one labeled loader.
    Returns (accuracy, macro_f1)."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits, _ = model(images)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1


def evaluate_source_domains(model, val_loaders: dict, device):
    """Returns ({domain: macro_f1}, mean_macro_f1) -- the checkpoint-selection
    signal required by the assignment. NOTE: caller must put the model back
    into train() + re-freeze BN (task2.models.backbone.freeze_bn_running_stats)
    before resuming training -- evaluate_domain leaves the model in eval()."""
    per_domain_f1 = {}
    for domain, loader in val_loaders.items():
        _, f1 = evaluate_domain(model, loader, device)
        per_domain_f1[domain] = f1
    mean_f1 = sum(per_domain_f1.values()) / len(per_domain_f1)
    return per_domain_f1, mean_f1
