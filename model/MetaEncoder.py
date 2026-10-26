import torch
from torch import nn
from configs import get_config
cfg = get_config()

class MetadataEncoder(nn.Module):
    def __init__(self, num_features, d: int = 32 ):
        super().__init__()
        self.metadata_encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, d),
        )
    def forward(self, x):
        return self.metadata_encoder (x)
