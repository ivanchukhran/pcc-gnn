from typing import Callable
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dir_path: str, split: str, classes: list | None = None, transform: Callable | None = None):
        self.dir_path = dir_path
        self.split = split
        self.classes = classes
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
