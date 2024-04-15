import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ray
import numpy as np
from dataclasses import dataclass

from utils import load_ply, save_ply

id_to_category = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02834778': 'bicycle', '02843684': 'birdhouse', '02871439': 'bookshelf',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'watercraft',
    '04554684': 'washer', '02858304': 'boat', '02992529': 'cellphone'
}

category_to_id = {v: k for k, v in id_to_category.items()}

@dataclass
class HyperPlane:
    normal: np.ndarray
    bias: float

    @staticmethod
    def from_3d_points(points: np.ndarray) -> 'HyperPlane':
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        return HyperPlane(normal, -np.dot(normal, points[0]))

    @staticmethod
    def random_3d() -> 'HyperPlane':
        return HyperPlane.from_3d_points(np.random.rand(3, 3))

    def __str__(self) -> str:
        return " ".join(str(x) for x in self.normal) + f" {self.bias}"


    def checkpoint(self, point: np.ndarray) -> bool:
        return np.sign(np.dot(point, self.normal) + self.bias)

def generate_split_sample(points: np.ndarray, min_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    while True:
        checkpoint = HyperPlane.random_3d().checkpoint(points) > 0
        points_above = points[~checkpoint]
        points_below = points[checkpoint]
        if not min_points:
            return points_above, points_below
        if (len(points_above) and len(points_below)) >= min_points:
            return points_above, points_below

@ray.remote
def generate_n_samples(filename: str, category: str,  dataset_path: str, num_samples: int = 4, min_points: int = 1024) -> None:
    """
    Generate n samples from a given point cloud file.

    :param filename: str, name of the file
    :param category: str, category identifier of the file (e.g. '02691156' for airplane)
    :param dataset_path: str, path to the dataset
    :param num_samples: int, number of samples to generate
    """
    points_path = os.path.join(dataset_path, category, filename)
    points, _, _ = load_ply(points_path)
    filename, format = filename.split('.')
    for _ in range(num_samples):
        existing, missing = generate_split_sample(points, min_points)
        save_ply(existing, os.path.join(dataset_path, 'slices', 'existing', id_to_category.get(category, ""), f'{filename}_{_}.{format}'))
        save_ply(missing, os.path.join(dataset_path, 'slices', 'missing', id_to_category.get(category, ""), f'{filename}_{_}.{format}'))

def generate_dataset(dataset_path: str, categories: list | None = None, num_samples: int = 4, min_points: int = 1024):
    """
    Generate samples from a dataset.

    :param dataset_path: str, path to the dataset
    :param categories: Optional[list], list of categories to generate samples
    :param num_samples: int, number of samples to generate
    :param min_points: int, minimum number of points to generate
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found")
    category_dirs = os.listdir(dataset_path) # contains the category ids
    for category_dir in category_dirs:
        if categories and category_dir not in categories:
            continue
        os.makedirs(os.path.join(dataset_path, 'slices', 'existing', id_to_category.get(category_dir, "")), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'slices', 'missing', id_to_category.get(category_dir, "")), exist_ok=True)
    ray.init(num_cpus=os.cpu_count())
    ray.get([generate_n_samples.remote(filename, category_dir, dataset_path, num_samples, min_points)
             for category_dir in category_dirs for filename in os.listdir(os.path.join(dataset_path, category_dir))])
    ray.shutdown()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate samples from a dataset')
    parser.add_argument('--cfg', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()
    return args

def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def main():
    import json

    logger = setup_logger()
    args = parse_args()
    logger.info(f"Reading config file {args.cfg}")
    if not os.path.exists(args.cfg):
        logger.error(f"Config file {args.cfg} not found")
        raise FileNotFoundError(f"Config file {args.cfg} not found")
    logger.info("Loading config file")
    with open(args.cfg, 'r') as f:
        config = json.load(f)
    if not config.keys():
        logger.error("Config file is empty")
        raise ValueError("Config file is empty")
    logger.info("Generating dataset")
    generate_dataset(**config)

if __name__ == '__main__':
    main()
