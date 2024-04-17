import torch
from torch import nn
import numpy as np


def rotation_matrix(degrees: float) -> torch.Tensor:
    theta = np.radians(degrees)
    rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]]).float()
    return rotation_matrix

def random_rotation_matrix() -> torch.Tensor:
    theta = np.random.rand() * 360
    return rotation_matrix(theta)
