import os
import sys
import torch
from torchvision import transforms
from torchvision.utils import save_image
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from task1.data.dataset import STL10Subset

def get_non_identity_permutation(num_patches, rng):
    """Ensures the generated permutation actually shuffles the image."""
    perm = list(range(num_patches))
    while True:
        rng.shuffle(perm)
        if perm != list(range(num_patches)):
            return perm

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # task1/
    out_dir = os.path.join(base_dir, 'data', 'patch_shuffle')
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = STL10Subset('test')
    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    
    rng = random.Random(6304)
    grid_size = 4
    num_patches = grid_size ** 2
    
    print(f"Generating 4x4 patch-shuffled images in {out_dir}...")
    
    for idx in tqdm(range(len(dataset))):
        img, label = dataset.base_dataset[dataset.indices[idx]]
        img_t = to_tensor(resize(img))
        
        _, H, W = img_t.shape
        patch_h = H // grid_size
        patch_w = W // grid_size
        
        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                patches.append(img_t[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w])
                
        # Generate the strict seeded permutation for this specific image
        perm = get_non_identity_permutation(num_patches, rng)
        shuffled_patches = [patches[p] for p in perm]
        
        # Reconstruct the physically scrambled image
        shuffled_img = torch.zeros_like(img_t)
        idx_patch = 0
        for i in range(grid_size):
            for j in range(grid_size):
                shuffled_img[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = shuffled_patches[idx_patch]
                idx_patch += 1
                
        # Save to disk to ensure identical reuse across all models
        out_name = f"{idx:04d}_label_{label}.png"
        save_image(shuffled_img, os.path.join(out_dir, out_name))
        
    print("Generation complete. The exact same images are saved for evaluation.")

if __name__ == "__main__":
    main()