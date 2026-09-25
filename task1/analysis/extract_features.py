import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import models
import open_clip
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from common.seed import set_seed
from task1.data.dataset import STL10Subset

def extract_and_save_features(model, dataloader, device, save_path, is_openclip=False):
    model.eval()
    features_list, labels_list = [], []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc=f"Extracting {os.path.basename(save_path)}"):
            imgs = imgs.to(device)
            
            if is_openclip:
                out = model.encode_image(imgs)
                # Ensure the CLIP image embedding is normalized as explicitly requested
                out = out / out.norm(dim=-1, keepdim=True)
            else:
                out = model(imgs)
                
            features_list.append(out.cpu())
            labels_list.append(lbls)
            
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    torch.save({'features': features, 'labels': labels}, save_path)

def main():
    set_seed(6304)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'features')
    os.makedirs(save_dir, exist_ok=True)

    # 1. ResNet-50 with IMAGENET1K_V2 (Updated)
    print("\nLoading ResNet-50...")
    resnet_weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=resnet_weights)
    resnet.fc = torch.nn.Identity() # Leaves the global-average-pooled feature (B, 2048)
    resnet = resnet.to(device)
    resnet_transform = resnet_weights.transforms()

    # 2. ViT-B/16 with IMAGENET1K_V1
    print("Loading ViT-B/16...")
    vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    vit = models.vit_b_16(weights=vit_weights)
    vit.heads = torch.nn.Identity() # Leaves the final class token (B, 768)
    vit = vit.to(device)
    vit_transform = vit_weights.transforms()

    # 3. OpenCLIP with 'openai' weights (Updated)
    print("Loading OpenCLIP...")
    clip_model, _, clip_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device)

    models_dict = {
        'resnet50': (resnet, resnet_transform, False),
        'vit_b_16': (vit, vit_transform, False),
        'openclip': (clip_model, clip_transform, True)
    }

    splits = ['train', 'val', 'test']
    
    for model_name, (model, transform, is_clip) in models_dict.items():
        print(f"\n--- Processing {model_name} ---")
        for split in splits:
            save_path = os.path.join(save_dir, f"{model_name}_{split}.pt")
            dataset = STL10Subset(split, transform=transform)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
            extract_and_save_features(model, dataloader, device, save_path, is_openclip=is_clip)

if __name__ == "__main__":
    main()