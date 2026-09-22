import os
import sys
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.seed import set_seed
from task1.dataset import STL10Subset

def main():
    set_seed(6304)
    dataset = STL10Subset('test')
    
    # Define our 5 specific class pairs
    pairs = [
        (5, 2, "dog", "car"),
        (3, 9, "cat", "truck"),
        (1, 0, "bird", "airplane"),
        (6, 8, "horse", "ship"),
        (7, 4, "monkey", "deer")
    ]
    
    # Create directories for the raw pairs
    base_out = os.path.join(os.path.dirname(__file__), 'data', 'cue_conflicts_raw')
    os.makedirs(base_out, exist_ok=True)
    
    # Extract images by class
    class_images = {i: [] for i in range(10)}
    for idx in range(len(dataset)):
        img, label = dataset.base_dataset[dataset.indices[idx]]
        class_images[label].append(transforms.Resize((224, 224))(img))
        
    generated_count = 0
    
    for class_a, class_b, name_a, name_b in pairs:
        # Direction 1: Shape=A, Texture=B (15 images)
        for i in range(30):
            shape_img = class_images[class_a][i]
            tex_img = class_images[class_b][i] # Use matching index for random texture pairing
            
            shape_path = os.path.join(base_out, f"{generated_count:03d}_shape_{name_a}_tex_{name_b}_content.jpg")
            tex_path = os.path.join(base_out, f"{generated_count:03d}_shape_{name_a}_tex_{name_b}_style.jpg")
            
            shape_img.save(shape_path)
            tex_img.save(tex_path)
            generated_count += 1
            
        # Direction 2: Shape=B, Texture=A (15 images)
        for i in range(20, 50):
            shape_img = class_images[class_b][i]
            tex_img = class_images[class_a][i]
            
            shape_path = os.path.join(base_out, f"{generated_count:03d}_shape_{name_b}_tex_{name_a}_content.jpg")
            tex_path = os.path.join(base_out, f"{generated_count:03d}_shape_{name_b}_tex_{name_a}_style.jpg")
            
            shape_img.save(shape_path)
            tex_img.save(tex_path)
            generated_count += 1

    print(f"Successfully prepared {generated_count} content/style pairs for AdaIN generation in {base_out}")

if __name__ == "__main__":
    main()