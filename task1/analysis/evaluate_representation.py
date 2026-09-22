import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as TF
import open_clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import glob
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.seed import set_seed
from task1.dataset import STL10Subset
from task1.color_bias import get_norm_transform

class InterventionDataset(Dataset):
    def __init__(self, transform_type='clean'):
        self.raw = STL10Subset('test')
        self.transform_type = transform_type
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self): return len(self.raw)
    
    def __getitem__(self, idx):
        img, label = self.raw.base_dataset[self.raw.indices[idx]]
        img_t = self.to_tensor(self.resize(img))
        
        if self.transform_type == 'grayscale':
            img_t = transforms.Grayscale(num_output_channels=3)(img_t)
        elif self.transform_type == 'hue':
            img_pil = transforms.ToPILImage()(img_t)
            img_pil = TF.adjust_hue(img_pil, 0.3)
            img_t = self.to_tensor(img_pil)
            
        return img_t, label

class TranslationDataset(Dataset):
    def __init__(self, delta=32, dx=32, dy=32):
        self.raw = STL10Subset('test')
        self.delta = delta
        self.dx = dx
        self.dy = dy
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.raw)

    def __getitem__(self, idx):
        img, label = self.raw.base_dataset[self.raw.indices[idx]]
        img_t = self.to_tensor(self.resize(img))
        padded = F.pad(img_t.unsqueeze(0), (self.delta, self.delta, self.delta, self.delta), mode='reflect').squeeze(0)
        start_x = self.delta - self.dx
        start_y = self.delta - self.dy
        return padded[:, start_y:start_y+224, start_x:start_x+224], label

class StylizedConflictDataset(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*_stylized.png")))
        self.indices = [int(os.path.basename(f).split('_')[0]) for f in self.files]
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.to_tensor(self.resize(img)), self.indices[idx]

class SavedPatchDataset(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.to_tensor(img), 0

def get_features(model, loader, norm, device, is_clip, return_indices=False):
    features = []
    all_indices = []
    with torch.no_grad():
        for batch in loader:
            imgs, indices = batch[0], batch[1]
            imgs = norm(imgs.to(device))
            feats = model.encode_image(imgs) if is_clip else model(imgs)
            feats = F.normalize(feats.view(feats.size(0), -1), p=2, dim=-1)
            features.append(feats.cpu().numpy())
            if return_indices:
                all_indices.extend(indices.numpy())
    feats_concat = np.concatenate(features, axis=0)
    if return_indices:
        return feats_concat, np.array(all_indices)
    return feats_concat

def main():
    set_seed(6304)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    shuffle_dir = os.path.join(base_dir, 'data', 'patch_shuffle')
    conflict_dir = os.path.join(base_dir, 'data', 'cue_conflicts_stylized')
    
    models_dict = {
        'resnet50': (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), False),
        'vit_b_16': (models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), False),
        'openclip': (open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')[0], True)
    }
    models_dict['resnet50'][0].fc = nn.Identity()
    models_dict['vit_b_16'][0].heads = nn.Identity()
    
    clean_loader = DataLoader(InterventionDataset('clean'), batch_size=64, shuffle=False)
    grayscale_loader = DataLoader(InterventionDataset('grayscale'), batch_size=64, shuffle=False)
    hue_loader = DataLoader(InterventionDataset('hue'), batch_size=64, shuffle=False)
    trans_loader = DataLoader(TranslationDataset(delta=32, dx=32, dy=32), batch_size=64, shuffle=False)
    conflict_loader = DataLoader(StylizedConflictDataset(conflict_dir), batch_size=64, shuffle=False)
    shuffle_loader = DataLoader(SavedPatchDataset(shuffle_dir), batch_size=64, shuffle=False)

    print("=" * 65)
    print("REPRESENTATION STABILITY & t-SNE (ALL INTERVENTIONS)")
    print("=" * 65)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('t-SNE Visualization: Clean vs. Patch-Shuffled Features', fontsize=16)

    for ax_idx, (m_name, (model, is_clip)) in enumerate(models_dict.items()):
        model = model.to(device).eval()
        norm = get_norm_transform(m_name)
        
        print(f"\n[{m_name.upper()}] Extracting features...")
        clean_feats = get_features(model, clean_loader, norm, device, is_clip)
        gray_feats = get_features(model, grayscale_loader, norm, device, is_clip)
        hue_feats = get_features(model, hue_loader, norm, device, is_clip)
        trans_feats = get_features(model, trans_loader, norm, device, is_clip)
        shuf_feats = get_features(model, shuffle_loader, norm, device, is_clip)
        
        conf_feats, conf_indices = get_features(model, conflict_loader, norm, device, is_clip, return_indices=True)
        
        sim_gray = np.mean(np.sum(clean_feats * gray_feats, axis=1))
        sim_hue = np.mean(np.sum(clean_feats * hue_feats, axis=1))
        sim_trans = np.mean(np.sum(clean_feats * trans_feats, axis=1))
        sim_shuf = np.mean(np.sum(clean_feats * shuf_feats, axis=1))
        
        clean_subset_for_conf = clean_feats[conf_indices]
        sim_conf = np.mean(np.sum(clean_subset_for_conf * conf_feats, axis=1))
        
        print(f"  Mean Cosine Similarity (Grayscale):   {sim_gray:.4f}")
        print(f"  Mean Cosine Similarity (Hue +0.3):    {sim_hue:.4f}")
        print(f"  Mean Cosine Similarity (Trans $\delta=32$): {sim_trans:.4f}")
        print(f"  Mean Cosine Similarity (Cue Conflict):{sim_conf:.4f}")
        print(f"  Mean Cosine Similarity (Shuffle):     {sim_shuf:.4f}")

        # Compute t-SNE for Clean vs Shuffle
        combined_feats = np.vstack((clean_feats, shuf_feats))
        tsne = TSNE(n_components=2, perplexity=30, random_state=6304, init='pca', learning_rate='auto')
        embeddings = tsne.fit_transform(combined_feats)
        
        n_clean = len(clean_feats)
        clean_emb = embeddings[:n_clean]
        shuf_emb = embeddings[n_clean:]
        
        ax = axes[ax_idx]
        ax.scatter(clean_emb[:, 0], clean_emb[:, 1], c='blue', label='Clean', alpha=0.5, s=15)
        ax.scatter(shuf_emb[:, 0], shuf_emb[:, 1], c='red', label='Shuffled', alpha=0.5, s=15)
        ax.set_title(f"{m_name.upper()}")
        ax.legend()
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'tsne_visualization.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n=> t-SNE plot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
    