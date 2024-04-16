from .base_dataset import BaseDataset

class ShapeNet(BaseDataset):
    def __init__(self, dir_path: str, split: str, classes: list | None = None, use_random_rotation: bool = False):
        super().__init__(dir_path, split, classes)
        self.use_random_rotation = use_random_rotation
