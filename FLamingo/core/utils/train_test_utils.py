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
            
            
if __name__ == '__main__':
    # Test infinite_dataloader
    test_data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_label = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    inf_loader = infinite_dataloader(test_dataloader)
    torch.manual_seed(0)
    for i in range(50):
        print(next(inf_loader))