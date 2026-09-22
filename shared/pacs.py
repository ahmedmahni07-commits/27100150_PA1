import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

# Pretrained ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms():
    """Returns the exact training and validation transforms specified for the protocol."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, eval_transform

def load_pacs_domain(root_dir: str, domain: str, transform=None) -> Dataset:
    """Loads a single PACS domain using standard ImageFolder structure."""
    domain_path = os.path.join(root_dir, domain)
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Dataset path not found: {domain_path}")
    
    return ImageFolder(root=domain_path, transform=transform)