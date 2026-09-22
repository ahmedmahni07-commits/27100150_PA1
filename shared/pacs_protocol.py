import json
import os
import itertools
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from shared.pacs import get_transforms, load_pacs_domain

SEED = 6304

def get_stratified_split(dataset, domain_name, save_dir="shared/splits"):
    """
    Creates or loads a reproducible 80/20 stratified split for a source domain.
    Saves the split indices to a JSON file to match the repository structure.
    """
    os.makedirs(save_dir, exist_ok=True)
    split_file = os.path.join(save_dir, f"pacs_{domain_name}_seed{SEED}.json")
    
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            indices = json.load(f)
            train_idx, val_idx = indices['train'], indices['val']
    else:
        # Extract targets for stratification
        targets = dataset.targets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        
        # Generator yields train/test indices
        train_idx, val_idx = next(sss.split(dataset.samples, targets))
        
        # Convert to lists for JSON serialization
        train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
        with open(split_file, 'w') as f:
            json.dump({'train': train_idx, 'val': val_idx}, f)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def setup_task2_dataloaders(data_root: str = "shared/pacs", num_workers: int = 0):
    """
    Initializes the dataloaders for Task 2 (UDA: Photo, Art, Cartoon -> Sketch).
    Source domains batch size = 8. Target domain batch size = 24.
    """
    train_transform, eval_transform = get_transforms()
    
    source_domains = ['photo', 'art_painting', 'cartoon']
    target_domain = 'sketch'
    
    train_loaders = {}
    val_loaders = {}
    
    # 1. Setup Source Domains (80/20 split)
    for domain in source_domains:
        # Load raw dataset without transforms first to get targets for splitting
        raw_dataset = load_pacs_domain(data_root, domain, transform=None)
        
        # Get stratified indices
        train_subset, val_subset = get_stratified_split(raw_dataset, domain)
        
        # Apply transforms by wrapping the subset's underlying dataset 
        # (PyTorch Subsets share the underlying dataset reference)
        train_subset.dataset.transform = train_transform
        
        # We need a separate instance for validation to apply eval_transform
        val_dataset = load_pacs_domain(data_root, domain, transform=eval_transform)
        val_subset = Subset(val_dataset, val_subset.indices)
        
        # Create loaders: Batch size 8 for sources
        train_loaders[domain] = DataLoader(train_subset, batch_size=8, shuffle=True, drop_last=True, num_workers=num_workers)
        val_loaders[domain] = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=num_workers)

    # 2. Setup Target Domain (100% available for adaptation without labels)
    target_dataset = load_pacs_domain(data_root, target_domain, transform=train_transform)
    target_loader = DataLoader(target_dataset, batch_size=24, shuffle=True, drop_last=True, num_workers=num_workers)
    
    # 3. Setup Target Evaluation (used only at the end)
    target_eval_dataset = load_pacs_domain(data_root, target_domain, transform=eval_transform)
    target_eval_loader = DataLoader(target_eval_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return train_loaders, val_loaders, target_loader, target_eval_loader

class MultiDomainIterator:
    """
    Combines multiple source loaders and one target loader.
    Cycles any loader that runs out of data before the longest loader finishes.
    """
    def __init__(self, source_loaders, target_loader):
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        
        # Calculate max iterations based on the largest dataset
        self.max_steps = max(
            [len(loader) for loader in source_loaders.values()] + [len(target_loader)]
        )

    def __iter__(self):
        # Create infinite cyclers for all loaders
        self.iters = {
            domain: itertools.cycle(loader) 
            for domain, loader in self.source_loaders.items()
        }
        self.target_iter = itertools.cycle(self.target_loader)
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step >= self.max_steps:
            raise StopIteration
            
        self.current_step += 1
        
        # Fetch 8 images from each source domain
        source_batches = {
            domain: next(iterator) 
            for domain, iterator in self.iters.items()
        }
        
        # Fetch 24 images from target domain
        target_batch = next(self.target_iter)
        
        return source_batches, target_batch