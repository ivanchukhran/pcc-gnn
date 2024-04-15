from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dir_path: str, split: str):
        self.dir_path = dir_path
        self.split = split

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
