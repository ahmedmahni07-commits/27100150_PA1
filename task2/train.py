import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

# Updated imports to use the refactored wrapper and plotting utility
from task2.methods.source_only import SourceOnly 
from task2.evaluation.metrics import evaluate_source_domains
from shared.pacs_protocol import setup_task2_dataloaders, MultiDomainIterator
from common.plotting import plot_training_curves

def set_bn_eval(module):
    """
    Forces BatchNorm layers into evaluation mode to freeze running mean/variance.
    Scale and bias (gamma/beta) remain trainable.
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def train_task2():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs("task2/results", exist_ok=True)
    
    # 1. Setup Data
    train_loaders, val_loaders, target_loader, _ = setup_task2_dataloaders(data_root="shared/pacs/images")

    # 2. Setup Model and Optimizer
    # Using the refactored SourceOnly class (Criterion is now handled internally)
    model = SourceOnly().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 3. Training Configurations & Tracking
    max_epochs = 30
    patience = 5
    best_mean_f1 = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    
    # Initialize history tracking for the plots
    history = {
        'loss': [], 
        'mean_f1': [], 
        'domains': {'photo': [], 'art_painting': [], 'cartoon': []}
    }

    print(f"Starting training on {device}...")

    for epoch in range(max_epochs):
        # Apply strict BatchNorm policy
        model.train()
        model.apply(set_bn_eval) 
        
        train_iter = MultiDomainIterator(train_loaders, target_loader)
        epoch_loss = 0.0
        steps = 0
        
        for source_batches, target_batch in train_iter:
            optimizer.zero_grad()
            
            # Unpack source batches (Photo, Art, Cartoon)
            # Concatenate all 24 source images and labels
            src_images = torch.cat([batch[0] for batch in source_batches.values()], dim=0).to(device)
            src_labels = torch.cat([batch[1] for batch in source_batches.values()], dim=0).to(device)
            
            # Unpack target batch (Sketch)
            tgt_images = target_batch[0].to(device)
            
            # Compute method-specific loss 
            # (Unified API: works seamlessly for SourceOnly, DAN, DANN, etc.)
            loss, step_metrics = model.compute_loss(src_images, src_labels, tgt_images)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_train_loss = epoch_loss / steps
        
        # 4. Validation & Checkpoint Selection (mean macro-F1 across 3 source domains)
        domain_f1_scores, mean_f1 = evaluate_source_domains(model, val_loaders, device)
        
        # Track metrics for plotting
        history['loss'].append(avg_train_loss)
        history['mean_f1'].append(mean_f1)
        for dom, f1 in domain_f1_scores.items():
            history['domains'][dom].append(f1)
        
        # Report individual domains before the pooled mean
        print(f"Epoch {epoch+1}/{max_epochs} - Loss: {avg_train_loss:.4f} - Mean Source F1: {mean_f1:.4f}")
        for dom, f1 in domain_f1_scores.items():
            print(f"  └ {dom.capitalize()}: {f1:.4f}")
            
        # 5. Early Stopping Logic
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, "task2/results/best_source_only.pth")
            print("  [*] Best checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(f"  [!] No improvement for {epochs_without_improvement} epoch(s).")
            
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print(f"Training completed. Best Mean Source F1: {best_mean_f1:.4f}")
    
    # 6. Generate and save the training curve
    plot_training_curves(history, save_path="report/figures/source_only_learning_curve.png")

if __name__ == "__main__":
    # Ensure reproducibility seed is set for PyTorch operations
    torch.manual_seed(6304)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(6304)
        
    train_task2()