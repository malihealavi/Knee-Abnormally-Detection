import torch
from torch import nn
from configs import get_config
cfg = get_config()

def stack_neighbors(volume: torch.Tensor, window: int = 1) -> torch.Tensor:
    """
    volume: (N, H, W)  — single modality, N slices
    returns: (N, 2*window+1, H, W)  — each slice paired with neighbors
    """
    N, H, W = volume.shape
    slices = []
    for i in range(N):
        neighbors = []
        for offset in range(-window, window + 1):
            idx = max(0, min(N - 1, i + offset))
            neighbors.append(volume[idx])
        slices.append(torch.stack(neighbors, dim=0))  # (3, H, W)
    return torch.stack(slices, dim=0)
