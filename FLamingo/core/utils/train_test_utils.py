import torch
from torch.utils.data import Dataset, DataLoader


def infinite_dataloader(dataloader):
    """
    Infinitely yield data from dataloader  
    
    Args:
        dataloader: DataLoader instance
    Returns:
        data: data from dataloader
    """
    while True:
        for data in dataloader:
            yield data