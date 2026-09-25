"""
Step 3: GCSC (Good Closed-Set Classifier) -- exactly the Vanilla recipe
(same model, initialization procedure, optimizer, schedule, batch size,
epochs, seed, and checkpoint rule) with ONE change: RandAugment(num_ops=2,
magnitude=9) inserted after crop+flip and before ToTensor/normalization.

That's a difference in TRAINING DATA AUGMENTATION only, not in the model
or training step -- so this module just re-exports vanilla's model/build/
training_step/checkpoint functions unchanged. task4/train.py is what
actually applies the different transform, by checking `cfg["method"] ==
"gcsc"` and calling task4.data.cifar10.get_gcsc_train_transform() instead
of get_transforms("train").
"""

from __future__ import annotations

from task4.methods.vanilla import (
    VanillaModel as GCSCModel,
    build_model,
    load_checkpoint,
    save_checkpoint,
    training_step,
)

__all__ = ["GCSCModel", "build_model", "training_step", "save_checkpoint", "load_checkpoint"]
