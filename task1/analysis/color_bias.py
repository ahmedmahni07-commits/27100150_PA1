import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as F_vision
import open_clip
from tqdm import tqdm
import copy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.seed import set_seed
from task1.dataset import STL10Subset

# --- Decoupled Normalization ---
def get_norm_transform(model_name):
    if model_name == 'openclip':
        return transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    else:
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --- Custom Dataset for Color Interventions ---
class ColorInterventionDataset(Dataset):
    def __init__(self, split_name, intervention_type):
        self.raw = STL10Subset(split_name)
        self.intervention = intervention_type
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224)) # Enforce the 224x224 assignment constraint
        
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        img, label = self.raw.base_dataset[self.raw.indices[idx]]
        img = self.resize(img) # Resize the raw PIL image first
        
        if self.intervention == 'clean':
            img_t = self.to_tensor(img)
        elif self.intervention == 'grayscale':
            img = img.convert("L").convert("RGB") # Remove color, keep 3 channels
            img_t = self.to_tensor(img)
        elif self.intervention == 'hue_rotation':
            img_t = self.to_tensor(img)
            # Shift hue by +0.3 (flips color spectrum predictably)
            img_t = F_vision.adjust_hue(img_t, 0.3)
            
        return img_t, label
    
# --- Helper: Train Linear Head ---
def train_head_in_memory(model_name, features_dir, device):
    set_seed(6304)
    train_data = torch.load(os.path.join(features_dir, f"{model_name}_train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(features_dir, f"{model_name}_val.pt"), weights_only=False)
    
    X_t = train_data['features'].view(train_data['features'].size(0), -1)
    y_t = train_data['labels']
    X_v = val_data['features'].view(val_data['features'].size(0), -1)
    y_v = val_data['labels']
    
    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=256, shuffle=False)
    
    head = nn.Linear(X_t.shape[1], 10).to(device)
    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_weights = None
    no_improve = 0
    
    for epoch in range(50):
        head.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(head(X.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                out = head(X.to(device))
                correct += (out.argmax(dim=1) == y.to(device)).sum().item()
                total += y.size(0)
        acc = correct / total
        
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(head.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5: break
            
    head.load_state_dict(best_weights)
    return head.eval()

# --- Execution ---
def main():
    set_seed(6304)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    
    models_dict = {
        'resnet50': (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), False),
        'vit_b_16': (models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), False),
        'openclip': (open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')[0], True)
    }
    models_dict['resnet50'][0].fc = nn.Identity()
    models_dict['vit_b_16'][0].heads = nn.Identity()
    
    interventions = ['clean', 'grayscale', 'hue_rotation']
    
    print("="*60)
    print("COLOR BIAS INTERVENTION RESULTS")
    print("="*60)
    
    for m_name, (model, is_clip) in models_dict.items():
        model = model.to(device).eval()
        head = train_head_in_memory(m_name, os.path.join(base_dir, 'features'), device)
        norm = get_norm_transform(m_name)
        
        predictions_map = {}
        
        for inv in interventions:
            loader = DataLoader(ColorInterventionDataset('test', inv), batch_size=64)
            preds, labels = [], []
            
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs = norm(imgs.to(device))
                    feats = model.encode_image(imgs) if is_clip else model(imgs)
                    if is_clip: feats = feats / feats.norm(dim=-1, keepdim=True)
                    batch_preds = head(feats.view(feats.size(0), -1)).argmax(dim=1)
                    
                    preds.extend(batch_preds.cpu().numpy())
                    labels.extend(lbls.numpy())
            
            acc = np.mean(np.array(preds) == np.array(labels)) * 100
            predictions_map[inv] = np.array(preds)
            
            if inv == 'clean':
                clean_acc = acc
                print(f"\n[{m_name.upper()}]")
            else:
                acc_change = acc - clean_acc
                consistency = np.mean(predictions_map[inv] == predictions_map['clean']) * 100
                print(f"  {inv.capitalize():<12} -> Acc Change: {acc_change:+.2f}% | Consistency: {consistency:.2f}%")

if __name__ == "__main__":
    main()