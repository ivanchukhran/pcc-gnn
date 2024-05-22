import torch
from torch import nn
from torch_geometric import nn as gnn

class EdgeConvLayer(nn.Module):
    """
    EdgeConvLayer for Point Completion Network on 3D Point Clouds

    Args:
        - in_channels: int - number of input channels
        - out_channels: int - number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(EdgeConvLayer, self).__init__()
        self.conv = gnn.EdgeConv(nn=nn.Linear(2 * in_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)

class GraphEncoder(nn.Module):
    """
    GraphEncoder for Point Completion Network on 3D Point Clouds

    Args:
        - blocks: list[int] - list of integers representing the number of channels in each block
        - k: int - number of nearest neighbors to consider

    """

    def __init__(self, blocks: list[int]):
        super(GraphEncoder, self).__init__()
        if not blocks:
            raise ValueError("The blocks list should not be empty.")
        self.blocks = blocks
        self.encoder_modules = self._build_encoder(blocks=blocks)

    def _build_encoder(self, blocks: list[int]) -> nn.ModuleList:
        encoder_modules = nn.ModuleList()
        for i in range(len(blocks) - 1):
            encoder_modules.append(
                EdgeConvLayer(blocks[i], blocks[i+1])
            )
        return encoder_modules

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the graph encoder

        Args:
            - : torch.tensor - input tensor of shape (num_nodes, in_channels)
            - edge_index: torch.tensor - edge index tensor of shape (2, num_edges)
        Return:
            - torch.tensor - output tensor of shape (num_nodes, out_channels)
        """
        for module in self.encoder_modules:
            x = module(x, edge_index)
        return gnn.global_max_pool(x, batch)

    def __repr__(self) -> str:
        return f"GraphEncoder(k={self.k}, blocks={self.blocks})"
