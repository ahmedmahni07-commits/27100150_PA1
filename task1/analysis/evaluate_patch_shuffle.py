import os
import sys
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import open_clip
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from task1.dataset import STL10Subset
from task1.color_bias import train_head_in_memory, get_norm_transform

class CleanDataset(Dataset):
    def __init__(self):
        self.raw = STL10Subset('test')
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self): return len(self.raw)
    def __getitem__(self, idx):
        img, label = self.raw.base_dataset[self.raw.indices[idx]]
        return self.to_tensor(self.resize(img)), label

class SavedPatchDataset(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        label = int(re.search(r'_label_(\d+)\.png', path).group(1))
        img = Image.open(path).convert('RGB')
        return self.to_tensor(img), label

def get_predictions(model, head, loader, norm, device, is_clip):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = norm(imgs.to(device))
            feats = model.encode_image(imgs) if is_clip else model(imgs)
            if is_clip:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            
            preds = head(feats.view(feats.size(0), -1)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            
    return np.array(all_preds), np.array(all_labels)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    shuffle_dir = os.path.join(base_dir, 'data', 'patch_shuffle')
    
    if not os.path.exists(shuffle_dir) or len(os.listdir(shuffle_dir)) == 0:
        print("Error: Run generate_patch_shuffle.py first to create the static images.")
        sys.exit(1)
        
    models_dict = {
        'resnet50': (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), False),
        'vit_b_16': (models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), False),
        'openclip': (open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')[0], True)
    }
    models_dict['resnet50'][0].fc = nn.Identity()
    models_dict['vit_b_16'][0].heads = nn.Identity()
    
    print("=" * 65)
    print("PATCH SHUFFLE (4x4) EVALUATION")
    print("=" * 65)

    clean_loader = DataLoader(CleanDataset(), batch_size=64, shuffle=False)
    shuffled_loader = DataLoader(SavedPatchDataset(shuffle_dir), batch_size=64, shuffle=False)

    for m_name, (model, is_clip) in models_dict.items():
        model = model.to(device).eval()
        head = train_head_in_memory(m_name, os.path.join(base_dir, 'features'), device)
        norm = get_norm_transform(m_name)
        
        clean_preds, _ = get_predictions(model, head, clean_loader, norm, device, is_clip)
        shuffle_preds, labels = get_predictions(model, head, shuffled_loader, norm, device, is_clip)
        
        clean_acc = np.mean(clean_preds == labels) * 100
        shuffle_acc = np.mean(shuffle_preds == labels) * 100
        consistency = np.mean(shuffle_preds == clean_preds) * 100
        
        print(f"\n[{m_name.upper()}]")
        print(f"  Clean Accuracy:    {clean_acc:.2f}%")
        print(f"  Shuffled Accuracy: {shuffle_acc:.2f}%")
        print(f"  Accuracy Drop:     {clean_acc - shuffle_acc:.2f}%")
        print(f"  Consistency:       {consistency:.2f}%")

if __name__ == "__main__":
    main()