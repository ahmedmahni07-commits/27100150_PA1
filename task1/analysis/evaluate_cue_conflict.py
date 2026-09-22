import os
import sys
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import open_clip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.seed import set_seed
from task1.color_bias import train_head_in_memory, get_norm_transform

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(STL10_CLASSES)}

class StylizedConflictDataset(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*_stylized.png")))
        # Regex to extract shape and texture classes from the filename
        self.pattern = re.compile(r".*_shape_([a-z]+)_tex_([a-z]+)_stylized\.png")
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)
        match = self.pattern.match(filename)
        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
        
        shape_name, tex_name = match.groups()
        shape_label = CLASS_TO_IDX[shape_name]
        tex_label = CLASS_TO_IDX[tex_name]

        img = Image.open(path).convert('RGB')
        img = self.resize(img)
        img_t = self.to_tensor(img)

        return img_t, shape_label, tex_label

def main():
    set_seed(6304)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    stylized_dir = os.path.join(base_dir, 'data', 'cue_conflicts_stylized')

    dataset = StylizedConflictDataset(stylized_dir)
    n_total = len(dataset)
    
    print("=" * 65)
    print("SHAPE VS. TEXTURE (CUE-CONFLICT) EVALUATION")
    print("=" * 65)
    print(f"Total valid conflicts evaluated: {n_total}")
    
    if n_total < 200:
        print("WARNING: You have fewer than 200 images. The rubric requires at least 200 valid conflicts.")

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    models_dict = {
        'resnet50': (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), False),
        'vit_b_16': (models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), False),
        'openclip': (open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')[0], True)
    }
    models_dict['resnet50'][0].fc = nn.Identity()
    models_dict['vit_b_16'][0].heads = nn.Identity()

    for m_name, (model, is_clip) in models_dict.items():
        model = model.to(device).eval()
        head = train_head_in_memory(m_name, os.path.join(base_dir, 'features'), device)
        norm = get_norm_transform(m_name)

        n_shape = 0
        n_texture = 0
        n_other = 0

        with torch.no_grad():
            for imgs, shape_lbls, tex_lbls in loader:
                imgs = norm(imgs.to(device))
                feats = model.encode_image(imgs) if is_clip else model(imgs)
                if is_clip:
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                
                preds = head(feats.view(feats.size(0), -1)).argmax(dim=1).cpu()

                for pred, s_lbl, t_lbl in zip(preds, shape_lbls, tex_lbls):
                    if pred.item() == s_lbl.item():
                        n_shape += 1
                    elif pred.item() == t_lbl.item():
                        n_texture += 1
                    else:
                        n_other += 1

        denom = n_shape + n_texture
        shape_bias = (n_shape / denom * 100) if denom > 0 else 0.0
        coverage = ((n_shape + n_texture) / n_total * 100) if n_total > 0 else 0.0

        print(f"\n[{m_name.upper()}]")
        print(f"  Decisions: N_shape={n_shape} | N_texture={n_texture} | N_other={n_other}")
        print(f"  Shape Bias: {shape_bias:.2f}%")
        print(f"  Coverage:   {coverage:.2f}%")

if __name__ == "__main__":
    main()