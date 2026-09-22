import os
import sys
import glob
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import open_clip
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from task1.color_bias import train_head_in_memory, get_norm_transform

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(STL10_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    stylized_dir = os.path.join(base_dir, 'data', 'cue_conflicts_stylized')
    
    # Grab the first 4 images to showcase in the report
    files = sorted(glob.glob(os.path.join(stylized_dir, "*_stylized.png")))[:4]
    if not files:
        print("Error: No stylized images found.")
        return
        
    pattern = re.compile(r".*_shape_([a-z]+)_tex_([a-z]+)_stylized\.png")
    
    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()

    models_dict = {
        'resnet50': (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), False),
        'vit_b_16': (models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), False),
        'openclip': (open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')[0], True)
    }
    models_dict['resnet50'][0].fc = nn.Identity()
    models_dict['vit_b_16'][0].heads = nn.Identity()

    # Load models and heads
    loaded_models = {}
    print("Loading models and extracting features...")
    for m_name, (model, is_clip) in models_dict.items():
        model = model.to(device).eval()
        head = train_head_in_memory(m_name, os.path.join(base_dir, 'features'), device)
        norm = get_norm_transform(m_name)
        loaded_models[m_name] = (model, head, norm, is_clip)

    fig, axes = plt.subplots(len(files), 2, figsize=(10, 3 * len(files)))
    fig.suptitle('Informative Cue-Conflict Case Studies', fontsize=16, y=1.02)

    with torch.no_grad():
        for i, path in enumerate(files):
            filename = os.path.basename(path)
            shape_name, tex_name = pattern.match(filename).groups()
            
            img_pil = Image.open(path).convert('RGB')
            img = to_tensor(resize(img_pil)).unsqueeze(0)
            
            preds_str = {}
            for m_name, (model, head, norm, is_clip) in loaded_models.items():
                img_norm = norm(img.to(device))
                feats = model.encode_image(img_norm) if is_clip else model(img_norm)
                if is_clip:
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                
                pred_idx = head(feats.view(feats.size(0), -1)).argmax(dim=1).item()
                pred_class = IDX_TO_CLASS[pred_idx]
                
                if pred_class == shape_name: tag = "(Shape)"
                elif pred_class == tex_name: tag = "(Texture)"
                else: tag = "(Other)"
                preds_str[m_name] = f"{pred_class.capitalize()} {tag}"

            # Plot Image (Left Column)
            ax_img = axes[i][0]
            ax_img.imshow(img_pil)
            ax_img.axis('off')
            
            # Plot Text Data (Right Column)
            ax_txt = axes[i][1]
            ax_txt.axis('off')
            
            text_content = (
                f"Ground Truth:\n"
                f"  • Shape: {shape_name.capitalize()}\n"
                f"  • Texture: {tex_name.capitalize()}\n\n"
                f"Model Predictions:\n"
                f"  • ResNet-50: {preds_str['resnet50']}\n"
                f"  • ViT-B/16: {preds_str['vit_b_16']}\n"
                f"  • OpenCLIP: {preds_str['openclip']}"
            )
            
            ax_txt.text(0.05, 0.5, text_content, fontsize=13, va='center', ha='left', 
                        bbox=dict(facecolor='#f8f9fa', alpha=1.0, edgecolor='#dee2e6', boxstyle='round,pad=1'))

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'cue_conflict_examples.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"\n=> Visualized table successfully saved to: {out_path}")

if __name__ == "__main__":
    main()