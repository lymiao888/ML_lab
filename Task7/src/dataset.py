import torch
from torch.utils.data import Dataset

class UnlabeledDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset  

    def __getitem__(self, index):
        img, _ = self.subset[index]
        return img

    def __len__(self):
        return len(self.subset)
