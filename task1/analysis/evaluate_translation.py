import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import open_clip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.seed import set_seed
from task1.dataset import STL10Subset
from task1.color_bias import train_head_in_memory, get_norm_transform

def reflect_pad_and_crop(img_t, dx, dy, delta):
    """
    Applies reflection padding of size delta on all sides, then crops a 224x224 window
    shifted by (dx, dy) to translate the object while preserving naturalistic edge statistics.
    """
    # img_t shape: [C, H, W] -> unsqueeze for F.pad which expects [N, C, H, W]
    padded = F.pad(img_t.unsqueeze(0), (delta, delta, delta, delta), mode='reflect').squeeze(0)
    
    # Original image is located at (delta, delta) in the padded tensor.
    # To shift the object by dx, dy, we shift the crop window in the OPPOSITE direction.
    start_x = delta - dx
    start_y = delta - dy
    
    return padded[:, start_y:start_y+224, start_x:start_x+224]

class TranslationDataset(Dataset):
    def __init__(self, split_name, delta=0, dx=0, dy=0):
        self.raw = STL10Subset(split_name)
        self.delta = delta
        self.dx = dx
        self.dy = dy
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        img, label = self.raw.base_dataset[self.raw.indices[idx]]
        img_t = self.to_tensor(self.resize(img))
        
        if self.delta > 0:
            img_t = reflect_pad_and_crop(img_t, self.dx, self.dy, self.delta)
            
        return img_t, label

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
    
    deltas = [0, 8, 16, 32]
    
    print("=" * 70)
    print("TRANSLATION EQUIVARIANCE (REFLECTION PAD + CROP)")
    print("=" * 70)

    plot_data = {m: {'acc': [], 'cons': []} for m in models_dict.keys()}

    for m_name, (model, is_clip) in models_dict.items():
        model = model.to(device).eval()
        head = train_head_in_memory(m_name, os.path.join(base_dir, 'features'), device)
        norm = get_norm_transform(m_name)
        
        print(f"\n[{m_name.upper()}]")
        
        # Base delta = 0
        clean_loader = DataLoader(TranslationDataset('test', delta=0), batch_size=64)
        clean_preds, labels = get_predictions(model, head, clean_loader, norm, device, is_clip)
        clean_acc = np.mean(clean_preds == labels) * 100
        
        print(f"  Delta  0 -> Acc: {clean_acc:.2f}% | Consistency: 100.00%")
        plot_data[m_name]['acc'].append(clean_acc)
        plot_data[m_name]['cons'].append(100.0)
        
        # Evaluate displacements
        for delta in deltas[1:]:
            dirs = [
                (delta, 0),   # Right
                (-delta, 0),  # Left
                (0, delta),   # Down
                (0, -delta)   # Up
            ]
            
            dir_accs = []
            dir_cons = []
            
            for dx, dy in dirs:
                loader = DataLoader(TranslationDataset('test', delta=delta, dx=dx, dy=dy), batch_size=64)
                preds, _ = get_predictions(model, head, loader, norm, device, is_clip)
                
                acc = np.mean(preds == labels) * 100
                cons = np.mean(preds == clean_preds) * 100
                
                dir_accs.append(acc)
                dir_cons.append(cons)
            
            # Average across the 4 cardinal directions
            avg_acc = np.mean(dir_accs)
            avg_cons = np.mean(dir_cons)
            
            plot_data[m_name]['acc'].append(avg_acc)
            plot_data[m_name]['cons'].append(avg_cons)
            
            print(f"  Delta {delta:2d} -> Acc: {avg_acc:.2f}% | Consistency: {avg_cons:.2f}%")

    # Generate Plots
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    for m_name in models_dict.keys():
        plt.plot(deltas, plot_data[m_name]['acc'], marker='o', label=m_name.upper())
    plt.title('Accuracy vs Displacement')
    plt.xlabel('Displacement (pixels)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(deltas)
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Consistency
    plt.subplot(1, 2, 2)
    for m_name in models_dict.keys():
        plt.plot(deltas, plot_data[m_name]['cons'], marker='o', label=m_name.upper())
    plt.title('Prediction Consistency vs Displacement')
    plt.xlabel('Displacement (pixels)')
    plt.ylabel('Consistency (%)')
    plt.xticks(deltas)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(base_dir, 'translation_curve.png')
    plt.savefig(plot_path)
    print(f"\nPlots successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()