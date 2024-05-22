import torch
from torch import nn

class Decoder(nn.Module):
    """
    Decoder for Point Completion Network on 3D Point Clouds. The decoder is used to decode the coarse point cloud
    to the fine point cloud.

    Args:
        - blocks: list[int] - list of integers representing the number of channels in each block
    """
    def __init__(self, blocks: list[int], grid_size: int = 4, grid_scale: float = 0.05):
        super(Decoder, self).__init__()
        self.blocks = blocks
        if grid_scale <= 0:
            raise ValueError("grid_scale should be greater than 0.")
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.decoder_modules = self._build_decoder(blocks=blocks)

    def _build_decoder(self, blocks: list[int]) -> nn.ModuleList:
        decoder_modules = nn.ModuleList()
        for i in range(len(blocks) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Conv1d(blocks[i], blocks[i+1], kernel_size=1),
                    nn.BatchNorm1d(blocks[i+1]),
                    nn.ReLU()
                )
            )
        return decoder_modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.decoder_modules:
            x = module(x)
        return x
