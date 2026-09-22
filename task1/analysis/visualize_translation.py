import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from task1.dataset import STL10Subset

def reflect_pad_and_crop(img_t, dx, dy, delta):
    padded = F.pad(img_t.unsqueeze(0), (delta, delta, delta, delta), mode='reflect').squeeze(0)
    start_x = delta - dx
    start_y = delta - dy
    return padded[:, start_y:start_y+224, start_x:start_x+224]

def main():
    base_dir = os.path.dirname(__file__)
    
    # Safely load the first image directly from the actual test set
    dataset = STL10Subset('test')
    img, _ = dataset.base_dataset[dataset.indices[0]] 
    img_t = transforms.ToTensor()(transforms.Resize((224, 224))(img))

    delta = 32
    shifts = {
        'Original': (0, 0),
        'Right (+32x)': (delta, 0),
        'Left (-32x)': (-delta, 0),
        'Down (+32y)': (0, delta),
        'Up (-32y)': (0, -delta)
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for ax, (title, (dx, dy)) in zip(axes, shifts.items()):
        if dx == 0 and dy == 0:
            shifted_t = img_t
        else:
            shifted_t = reflect_pad_and_crop(img_t, dx, dy, delta)
            
        ax.imshow(shifted_t.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'translation_example.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Translation visualizer saved to: {out_path}")

if __name__ == "__main__":
    main()