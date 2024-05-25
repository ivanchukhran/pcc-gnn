import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ray
import numpy as np
from dataclasses import dataclass

from utils import load_ply, save_ply

def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

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
    """
    Generate a split sample from a given point cloud.

    :param points: np.ndarray, point cloud
    :param min_points: int, minimum number of points to generate

    :return: tuple[np.ndarray, np.ndarray], points above the hyperplane, points below the hyperplane
    """
    while True:
        checkpoint = HyperPlane.random_3d().checkpoint(points) > 0
        points_above = points[~checkpoint]
        points_below = points[checkpoint]
        if len(points_above) == min_points:
            return points_above, points_below
        if len(points_below) == min_points:
            return points_below, points_above

def upsample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Upsample a point cloud to a given number of points.

    :param points: np.ndarray, point cloud
    :param num_points: int, number of points to upsample to

    :return: np.ndarray, upsampled point cloud
    """
    return np.concatenate([points, np.random.rand(num_points - len(points), 3)])

def downsample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Downsample a point cloud to a given number of points.

    :param points: np.ndarray, point cloud
    :param num_points: int, number of points to downsample to

    :return: np.ndarray, downsampled point cloud
    """
    return points[np.random.choice(len(points), num_points, replace=False)]

def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Sample a point cloud to a given number of points.

    :param points: np.ndarray, point cloud
    :param num_points: int, number of points to sample to

    :return: np.ndarray, sampled point cloud
    """
    if len(points) < num_points:
        return upsample_points(points, num_points)
    if len(points) > num_points:
        return downsample_points(points, num_points)
    return points

@ray.remote
def generate_n_samples(filename: str, category: str,  dataset_path: str, num_samples: int = 4, min_points: int = 1024) -> None:
    """
    Generate n samples from a given point cloud file.

    :param filename: str, name of the file
    :param category: str, category identifier of the file (e.g. '02691156' for airplane)
    :param dataset_path: str, path to the dataset
    :param num_samples: int, number of samples to generate
    """
    try:
        points_path = os.path.join(dataset_path, category, filename)
        points, _, _ = load_ply(points_path)
        points = sample_points(points, 2048)
        filename, format = filename.split('.')
        logger.info(f"Generating samples for {filename}")
        for _ in range(num_samples):
            existing, missing = generate_split_sample(points, min_points)
            save_ply(existing, os.path.join(dataset_path, 'slices', 'existing', category, f'{filename}_{_}.{format}'))
            save_ply(missing, os.path.join(dataset_path, 'slices', 'missing', category, f'{filename}_{_}.{format}'))
    except Exception as e:
        logger.error(f"Error generating samples for {filename}: {e}")
        return

def generate_dataset(dataset_path: str, classes: list | None = None, num_samples: int = 4, min_points: int = 1024, *args, **kwargs):
    """
    Generate samples from a dataset.

    :param dataset_path: str, path to the dataset
    :param categories: Optional[list], list of categories to generate samples
    :param num_samples: int, number of samples to generate
    :param min_points: int, minimum number of points to generate

    :return: None
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found")
    category_dirs = os.listdir(dataset_path) # contains the category ids
    if not classes:
        classes = category_dirs
    for category_dir in category_dirs:
        if category_dir not in classes:
            continue
        os.makedirs(os.path.join(dataset_path, 'slices', 'existing', category_dir), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'slices', 'missing', category_dir), exist_ok=True)
    logger.info(f"Setting up ray with {os.cpu_count()} CPUs")
    ray.init(num_cpus=os.cpu_count())
    logger.info("Generating samples")
    ray.get([generate_n_samples.remote(filename, category_dir, dataset_path, num_samples, min_points)
             for category_dir in category_dirs for filename in os.listdir(os.path.join(dataset_path, category_dir))])
    logger.info("Shutting down ray")
    ray.shutdown()
    logger.info("Samples generated")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate samples from a dataset')
    parser.add_argument('--cfg', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()
    return args

def generate_train_validation_test_split(dataset_path: str, classes: list | None = None, train: float = 0.8, validation: float = 0.1, test: float = 0.1, *args, **kwargs):
    """
    Generate a train, validation, test split from a dataset.

    :param dataset_path: str, path to the dataset
    :param classes: Optional[list], list of classes to generate the split
    :param train: float, percentage of the dataset to use for training
    :param validation: float, percentage of the dataset to use for validation
    :param test: float, percentage of the dataset to use for testing

    :return: None
    """
    if not os.path.exists(dataset_path):
        log_string = f"Dataset path {dataset_path} not found"
        logger.error(log_string)
        raise FileNotFoundError(log_string)
    category_dirs = os.listdir(dataset_path) # contains the category ids
    logger.info("Setting up splits directory")
    os.makedirs(os.path.join(dataset_path, 'slices'), exist_ok=True)

    if not classes:
        classes = category_dirs

    train_files, validation_files, test_files = [], [], []
    for category_dir in category_dirs:
        if category_dir not in classes:
            continue
        category_files = os.listdir(os.path.join(dataset_path, category_dir))
        train_split = int(len(category_files) * train)
        validation_split = int(len(category_files) * validation)
        test_split = int(len(category_files) * test)
        if (train_split + validation_split + test_split) != len(category_files):
            train_split += len(category_files) - (train_split + validation_split + test_split)
        np.random.shuffle(category_files)
        for i, filename in enumerate(category_files):
            if i < train_split:
                train_files.append(os.path.join(category_dir, filename))
            elif i < train_split + validation_split:
                validation_files.append(os.path.join(category_dir, filename))
            else:
                test_files.append(os.path.join(category_dir, filename))
    splits = {'train': train_files, 'validation': validation_files, 'test': test_files}
    for split, filelist in splits.items():
        logger.info(f"Writing {split} split")
        os.makedirs(os.path.join(dataset_path, 'slices', 'splits'), exist_ok=True)
        for filename in filelist:
            logger.info(f"Writing {filename} to {split}.list")
            with open(os.path.join(dataset_path, 'slices', 'splits', f'{split}.list'), 'a') as f:
                f.write(filename + '\n')


def main():
    import json
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
    logger.info("Generating train, validation, test split")
    generate_train_validation_test_split(**config)

if __name__ == '__main__':
    main()