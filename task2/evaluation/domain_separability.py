"""
Domain-separability diagnostic (Task 2 "Common Evaluation and Alignment
Diagnostic"): freeze the backbone, collect equal numbers of source-validation
and target features, and see how well a simple linear classifier can still
tell them apart. 50% = chance = domains indistinguishable; higher = more
residual domain information survives in the representation.

Run this only on a FIXED checkpoint, after training/model-selection are done
-- it's a diagnostic, not a training-time signal.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


@torch.no_grad()
def extract_features(model, loader, device, has_labels):
    model.eval()
    feats = []
    for batch in loader:
        images = batch[0] if has_labels else batch
        images = images.to(device)
        _, f = model(images)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


def compute_domain_separability(model, source_val_loaders: dict, target_loader, device):
    """source_val_loaders: {domain: DataLoader} (labeled, eval transform,
    e.g. the val_loaders returned by build_dataloaders in train.py).
    target_loader: DataLoader over the target domain (return_label=False is
    fine here -- only domain identity, never class label, is used)."""
    source_feats = np.concatenate(
        [extract_features(model, loader, device, has_labels=True)
         for loader in source_val_loaders.values()],
        axis=0,
    )
    target_feats = extract_features(model, target_loader, device, has_labels=False)

    rng = np.random.RandomState(SEED)
    n = min(len(source_feats), len(target_feats))  # equal numbers of each, per spec
    source_idx = rng.choice(len(source_feats), size=n, replace=False)
    target_idx = rng.choice(len(target_feats), size=n, replace=False)
    source_feats = source_feats[source_idx]
    target_feats = target_feats[target_idx]

    X = np.concatenate([source_feats, target_feats], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])  # 0 = source, 1 = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y,
    )
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)  # the domain-separability score