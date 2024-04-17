import os
from typing import Callable
import torch
from torch import nn
from utils import load_ply
from utils.transformations import random_rotation_matrix
from .base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    def __init__(self,
                 dir_path: str,
                 split: str,
                 classes: list | None = None,
                 use_random_rotation: bool = False,
                 num_samples: int = 4,
                 transform: Callable | None = None):
        super().__init__(dir_path, split, classes, transform)
        self.use_random_rotation = use_random_rotation
        self.num_samples = num_samples
        if not self.classes:
            self.classes = os.listdir(self.dir_path)
        split_file = os.path.join(self.dir_path, 'slices', 'splits', f"{split}.list")
        with open(split_file, "r") as f:
            files = f.read().splitlines()
        self.filenames = [file for file in files if file.split("/")[0] in self.classes]

    def __len__(self):
        return len(self.filenames) * self.num_samples

    def __getitem__(self, idx: int):
        print(idx)
        filename = self.filenames[idx // self.num_samples]
        text, format = filename.split('.')
        existing_filename = os.path.join(self.dir_path, 'slices', 'existing', f'{text}_{idx % self.num_samples}.{format}')
        missing_filename = os.path.join(self.dir_path, 'slices', 'missing', f'{text}_{idx % self.num_samples}.{format}')
        existing, _, _ = load_ply(existing_filename)
        missing, _, _ = load_ply(missing_filename)
        existing = torch.from_numpy(existing)
        missing = torch.from_numpy(missing)
        if self.use_random_rotation:
            rotation_matrix = random_rotation_matrix()
            existing = torch.matmul(existing, rotation_matrix)
            missing = torch.matmul(missing, rotation_matrix)
        if self.transform is not None:
            existing = self.transform(existing)
            missing = self.transform(missing)
        return {
            "existing": existing,
            "missing": missing
        }
