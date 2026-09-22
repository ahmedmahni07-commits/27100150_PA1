import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import open_clip
import copy
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.seed import set_seed

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def load_split(features_dir, model_name, split):
    path = os.path.join(features_dir, f"{model_name}_{split}.pt")
    data = torch.load(path, weights_only=False)
    X = data['features'].view(data['features'].size(0), -1)
    y = data['labels']
    return X, y

def train_linear_head(model_name, features_dir, device):
    set_seed(6304)

    X_train, y_train = load_split(features_dir, model_name, 'train')
    X_val, y_val = load_split(features_dir, model_name, 'val')
    X_test, y_test = load_split(features_dir, model_name, 'test')

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    in_features = X_train.shape[1]
    head = nn.Linear(in_features, 10).to(device)
    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    best_weights = None

    print(f"\nTraining Linear Head for {model_name}...")
    for epoch in range(50):
        head.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = head(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = head(X_b)
                correct += (out.argmax(dim=1) == y_b).sum().item()
                total += y_b.size(0)
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(head.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Testing phase
    head.load_state_dict(best_weights)
    head.eval()
    all_preds, all_labels, all_confs = [], [], []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = head(X_b)
            probs = F.softmax(logits, dim=1)
            
            confs, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            
    # Calculate required metrics
    import numpy as np
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    mean_conf = np.mean(all_confs)
    
    print(f"[{model_name.upper()}] Accuracy: {acc * 100:.2f}% | Macro-F1: {macro_f1:.4f} | Mean Max Conf: {mean_conf:.4f}")

def evaluate_clip_zeroshot(features_dir, device):
    print("\nEvaluating CLIP Zero-Shot...")
    X_test, y_test = load_split(features_dir, 'openclip', 'test')
    X_test, y_test = X_test.to(device), y_test.to(device)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device)
    clip_model.eval()

    prompts = [f"a photo of a {c}." for c in STL10_CLASSES]
    text_tokens = open_clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Scale logits using CLIP's learned logit_scale
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * (X_test @ text_features.T)
        
        # Softmax over scaled similarities as required
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
        
        acc = (preds == y_test).float().mean().item()
        macro_f1 = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average='macro')
        mean_conf = confs.mean().item()
        
        print(f"[OPENCLIP ZERO-SHOT] Accuracy: {acc * 100:.2f}% | Macro-F1: {macro_f1:.4f} | Mean Max Conf: {mean_conf:.4f}")

def main():
    set_seed(6304)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    features_dir = os.path.join(os.path.dirname(__file__), 'features')
    
    models = ['resnet50', 'vit_b_16', 'openclip']
    for m in models:
        train_linear_head(m, features_dir, device)
        
    evaluate_clip_zeroshot(features_dir, device)

if __name__ == "__main__":
    main()