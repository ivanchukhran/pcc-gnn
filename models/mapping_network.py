import torch
from torch import nn

class MappingNetwork(nn.Module):
    """
    MappingNetwork for Point Completion Network on 3D Point Clouds. The mapping network is used to map the latent
    representation to the coarse point cloud.

    Args:
        - latent_dim: int - dimension of the latent representation
        - n_blocks: int - number of blocks in the mapping network
        - num_coarse: int - number of points in the coarse point cloud
    """
    def __init__(self, latent_dim: int, n_blocks: int, num_coarse: int):
        super(MappingNetwork, self).__init__()
        if n_blocks < 1:
            raise ValueError("The number of blocks should be at least 1.")
        self.num_coarse = num_coarse
        self.mappings = self._build_mappings(blocks=[latent_dim] * n_blocks)

    def _build_mappings(self, blocks: list[int]) -> nn.Module:
        mappings_ = nn.Sequential()
        for i in range(len(blocks) - 1):
            mappings_.add_module(f"linear_{i}", nn.Linear(blocks[i], blocks[i+1]))
            mappings_.add_module(f"relu_{i}", nn.ReLU())
        mappings_.add_module(f"linear_{len(blocks)}", nn.Linear(blocks[-1], self.num_coarse * 3))
        return mappings_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mappings(x)
