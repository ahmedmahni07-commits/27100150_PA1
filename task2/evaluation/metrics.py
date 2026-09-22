import torch
from sklearn.metrics import f1_score

@torch.no_grad()
def evaluate_source_domains(model, val_loaders, device):
    """
    Evaluates the model on all source validation sets.
    Returns a dictionary of macro-F1 scores per domain and the mean macro-F1.
    """
    model.eval()
    domain_f1_scores = {}
    
    for domain, loader in val_loaders.items():
        all_preds = []
        all_targets = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images) # Assuming model returns (logits, features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        domain_f1_scores[domain] = macro_f1
        
    mean_f1 = sum(domain_f1_scores.values()) / len(domain_f1_scores)
    return domain_f1_scores, mean_f1