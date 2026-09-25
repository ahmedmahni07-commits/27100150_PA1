"""
Task 3's model-selection rule: EVERY method (ERM, DAN-DG, SAM) selects its
checkpoint using mean macro-F1 over the three SOURCE validation splits
only. Sketch is never touched by this module, or by anything upstream of
it (see common.pacs_protocol.source_balanced_batches) -- selecting on the
unseen target domain would break the Domain Generalization protocol.

This mirrors task2/train.py's inline selection logic exactly (same metric,
same early-stopping rule) but is broken out into its own module here per
Task 3's suggested repo layout, and to make the "source-only" constraint
explicit and easy to audit independently of the training loop.

Reuses task2.evaluation.metrics.evaluate_loader directly rather than
reimplementing it -- it's a generic function over any model exposing
.backbone/.head/.eval_mode() and any DataLoader, with no Task-2-specific
assumptions baked in.
"""

from __future__ import annotations

from task2.evaluation.metrics import evaluate_loader


def evaluate_all_source_domains(model, val_loaders: dict, device) -> dict:
    """
    val_loaders: {domain_name: DataLoader over that domain's val split},
    built only from build_or_load_source_splits -- no Sketch key can ever
    appear here by construction.

    Returns {domain_name: {"accuracy": ..., "macro_f1": ...}}.
    """
    per_domain = {}
    for domain, loader in val_loaders.items():
        result = evaluate_loader(model, loader, device)
        per_domain[domain] = {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
    return per_domain


def mean_source_macro_f1(per_domain_val: dict) -> float:
    return sum(v["macro_f1"] for v in per_domain_val.values()) / len(per_domain_val)


class SourceValidationSelector:
    """
    Stateful best-checkpoint tracker + early-stopping counter, driven only
    by mean_source_macro_f1. Usage inside an epoch loop:

        selector = SourceValidationSelector(patience=cfg["optim"]["early_stopping_patience"])
        ...
        per_domain_val = evaluate_all_source_domains(model, val_loaders, device)
        score = mean_source_macro_f1(per_domain_val)
        if selector.update(score):
            method_module.save_checkpoint(model, best_ckpt_path)
        elif selector.should_stop():
            break
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = -1.0
        self.epochs_without_improvement = 0

    def update(self, score: float) -> bool:
        """Returns True iff `score` is a new best (and resets the patience counter)."""
        if score > self.best_score:
            self.best_score = score
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience
