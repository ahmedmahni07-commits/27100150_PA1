import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18_backbone():
    """
    Loads ResNet18 with ImageNet V1 weights and removes the classification head.
    Returns the model and the output feature dimension (512 for ResNet18).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    
    # Store the input feature size of the original fully connected layer
    feature_dim = model.fc.in_features
    
    # Replace the classification head with an Identity layer
    model.fc = nn.Identity()
    
    return model, feature_dim