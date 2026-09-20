import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets

class STL10Subset(Dataset):
    def __init__(self, split_name, transform=None):
        """
        Args:
            split_name (str): 'train', 'val', or 'test'.
            transform (callable, optional): Image transformations to apply.
        """
        self.split_name = split_name
        self.transform = transform
        
        # 1. Define paths dynamically based on the location of this script
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
        splits_path = os.path.join(os.path.dirname(__file__), 'data', 'splits', 'stl10_splits.pt')
        
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"Splits file not found at {splits_path}. Run make_subset.py first.")
            
        # 2. Load the base STL-10 dataset directly from the raw folder (download=False)
        # Train and Val come from the base 'train' split; Test comes from the base 'test' split.
        base_split = 'train' if split_name in ['train', 'val'] else 'test'
        self.base_dataset = datasets.STL10(root=data_dir, split=base_split, download=False)
        
        # 3. Load the frozen indices we generated
        splits = torch.load(splits_path, weights_only=False)        
        if split_name == 'train':
            self.indices = splits['train_indices']
        elif split_name == 'val':
            self.indices = splits['val_indices']
        elif split_name == 'test':
            self.indices = splits['test_subset_indices']
        else:
            raise ValueError("split_name must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve the original dataset index mapping
        original_idx = self.indices[idx]
        
        # Fetch the PIL image and label from torchvision's STL10
        image, label = self.base_dataset[original_idx]
        
        # Apply the specific model's preprocessing transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label