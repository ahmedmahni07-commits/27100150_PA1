import torch.nn as nn

class PACSClassifier(nn.Module):
    """
    A 7-class linear classifier head for the PACS dataset.
    """
    def __init__(self, in_features: int, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.fc(x)