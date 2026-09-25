import os
import sys
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Add the cloned naoto0804 repository to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # task1/
adain_repo = os.path.join(base_dir, 'pytorch-AdaIN')
sys.path.append(adain_repo)

try:
    import net
    from function import adaptive_instance_normalization
except ImportError:
    print("Error: Could not import from pytorch-AdaIN. Ensure it is cloned inside the task1 directory.")
    sys.exit(1)

def download_weights(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    # Updated to the permanent GitHub Releases mirror
    vgg_url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
    dec_url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
    
    vgg_path = os.path.join(models_dir, "vgg_normalised.pth")
    dec_path = os.path.join(models_dir, "decoder.pth")
    
    if not os.path.exists(vgg_path):
        print("Downloading VGG encoder weights (GitHub)...")
        torch.hub.download_url_to_file(vgg_url, vgg_path)
    if not os.path.exists(dec_path):
        print("Downloading AdaIN decoder weights (GitHub)...")
        torch.hub.download_url_to_file(dec_url, dec_path)
    return vgg_path, dec_path

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    models_dir = os.path.join(adain_repo, 'models')
    vgg_path, dec_path = download_weights(models_dir)

    # Initialize naoto0804 architecture
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(dec_path, weights_only=False, map_location='cpu'))
    vgg.load_state_dict(torch.load(vgg_path, weights_only=False, map_location='cpu'))
    
    # Extract up to relu4_1 as required by AdaIN
    vgg = nn.Sequential(*list(vgg.children())[:31]) 
    
    decoder.to(device)
    vgg.to(device)

    raw_dir = os.path.join(base_dir, 'data', 'cue_conflicts_raw')
    out_dir = os.path.join(base_dir, 'data', 'cue_conflicts_stylized')
    os.makedirs(out_dir, exist_ok=True)

    content_files = sorted(glob.glob(os.path.join(raw_dir, "*_content.jpg")))
    
    # Function to resize and preserve aspect ratio exactly as naoto0804 does
    def transform(img):
        return transforms.ToTensor()(img).unsqueeze(0).to(device)

    print(f"Stylizing {len(content_files)} pairs using naoto0804/pytorch-AdaIN...")
    
    for c_path in tqdm(content_files):
        prefix = c_path.replace("_content.jpg", "")
        s_path = prefix + "_style.jpg"
        
        if not os.path.exists(s_path): continue
        
        c_img = transform(Image.open(c_path).convert('RGB'))
        s_img = transform(Image.open(s_path).convert('RGB'))

        with torch.no_grad():
            c_feat = vgg(c_img)
            s_feat = vgg(s_img)
            target_feat = adaptive_instance_normalization(c_feat, s_feat)
            out_img = decoder(target_feat)

        out_name = os.path.basename(prefix) + "_stylized.png"
        save_image(out_img.clamp(0, 1), os.path.join(out_dir, out_name))

    print(f"All stylized images generated in {out_dir}")

if __name__ == "__main__":
    main()