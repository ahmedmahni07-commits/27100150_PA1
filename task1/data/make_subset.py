import os
import sys
import torch
import numpy as np
from torchvision import datasets

# Temporarily add the root directory to the path so we can import common.seed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from common.seed import set_seed

def generate_stl10_splits():
    # 1. Lock the global seed to 6304 as required by the assignment
    set_seed(6304)
    
    # 2. Point directly to the raw folder where stl10_binary.tar.gz is located
    data_dir = os.path.join(os.path.dirname(__file__), 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Looking for stl10_binary.tar.gz in: {data_dir} ...")
    
    # 3. Use the official torchvision loader
    # It will find the .tar.gz file, extract it to a 'stl10_binary' folder, and load the dataset
    train_data = datasets.STL10(root=data_dir, split='train', download=True)
    test_data = datasets.STL10(root=data_dir, split='test', download=True)

    # 4. Create a stratified 80/20 Train/Val split
    train_targets = np.array(train_data.labels)
    train_indices = np.arange(len(train_targets))
    
    train_idx, val_idx = [], []
    for class_label in range(10): 
        class_specific_idx = train_indices[train_targets == class_label]
        np.random.shuffle(class_specific_idx)
        
        split_point = int(len(class_specific_idx) * 0.8)
        train_idx.extend(class_specific_idx[:split_point])
        val_idx.extend(class_specific_idx[split_point:])

    # 5. Create a class-balanced subset of 500 images
    test_targets = np.array(test_data.labels)
    test_indices = np.arange(len(test_targets))
    
    subset_test_idx = []
    for class_label in range(10):
        class_specific_idx = test_indices[test_targets == class_label]
        selected = np.random.choice(class_specific_idx, 50, replace=False)
        subset_test_idx.extend(selected)
        
    subset_test_idx = np.array(subset_test_idx)
    np.random.shuffle(subset_test_idx) 

    # 6. Save the indices to disk to freeze the evaluation subset
    save_dir = os.path.join(os.path.dirname(__file__), 'splits')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'stl10_splits.pt')
    torch.save({
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_subset_indices': subset_test_idx
    }, save_path)
    
    print(f"Success! Saved splits to: {save_path}")
    print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test subset size: {len(subset_test_idx)}")

if __name__ == "__main__":
    generate_stl10_splits()