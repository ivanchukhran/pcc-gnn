from functools import reduce
import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.nn import knn_graph


def encoder_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2 * in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU()
    )

def decoder_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU()
    )

class GraphEncoder(nn.Module):
    """
    GraphEncoder for Point Completion Network on 3D Point Clouds

    Parameters:

    :param cfg: dict - configuration dictionary. Contains the following keys:
        - k: int - number of nearest neighbors to consider for each node
        - n_blocks: int - number of encoder blocks
        - embedding_dim: int - dimension of the output embedding
        - in_channels: int - number of input channels
        - out_channels: int - number of output channels
        - conv: str - type of convolutional layer to use (EdgeConv or DynamicEdgeConv)

    """

    def __init__(self, blocks: list[int], k: int = 20, conv_type: str = 'EdgeConv'):
        super(GraphEncoder, self).__init__()
        self.k = k
        if not blocks:
            raise ValueError("The blocks list should not be empty.")
        self.blocks = blocks
        self.conv_type = conv_type
        self.encoder_modules = self._build_encoder(blocks=blocks)

    def _build_encoder(self, blocks: list[int]) -> nn.ModuleList:
        encoder_modules = nn.ModuleList()
        for i in range(len(blocks) - 1):
            encoder_modules.append(
                getattr(gnn, self.conv_type)(
                    encoder_block(blocks[i], blocks[i+1])
                )
            )
        return encoder_modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph encoder

        :param x: torch.tensor - input tensor of shape (num_nodes, in_channels)
        :param edge_index: torch.tensor - edge index tensor of shape (2, num_edges)
        :return: torch.tensor - output tensor of shape (num_nodes, out_channels)
        """
        edge_index = knn_graph(x, self.k, batch=None, loop=False)
        for module in self.encoder_modules:
            x = module(x, edge_index)
        return x

    def __repr__(self) -> str:
        return f"GraphEncoder(k={self.k}, blocks={self.blocks})"

class MLP(nn.Module):
    def __init__(self, latent_dim: int, n_blocks: int, num_coarse: int):
        super(MLP, self).__init__()
        if n_blocks < 1:
            raise ValueError("The number of blocks should be at least 1.")
        self.mlp = self._build_mlp(blocks=[latent_dim] * n_blocks, num_coarse=num_coarse)

    def _build_mlp(self, blocks: list[int], num_coarse: int) -> nn.Module:
        mlp = nn.Sequential()
        for i in range(len(blocks) - 1):
            mlp.add_module(f"linear_{i}", nn.Linear(blocks[i], blocks[i+1]))
            mlp.add_module(f"relu_{i}", nn.ReLU())
        mlp.add_module(f"linear_{len(blocks)}", nn.Linear(blocks[-1], num_coarse * 3))
        return mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)



class Decoder(nn.Module):
    def __init__(self, blocks: list[int]):
        super(Decoder, self).__init__()
        self.blocks = blocks
        self.decoder_modules = self._build_decoder(blocks=blocks)

    def _build_decoder(self, blocks: list[int]) -> nn.ModuleList:
        decoder_modules = nn.ModuleList()
        for i in range(len(blocks) - 1):
            decoder_modules.append(
                decoder_block(blocks[i], blocks[i+1])
            )
        return decoder_modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

if __name__ == '__main__':
    encoder = GraphEncoder(blocks=[3, 64, 128, 512, 1024], k=20, conv_type='EdgeConv')
    mlp = MLP(latent_dim=1024, n_blocks=3, num_coarse=2048)
    decoder = Decoder(blocks=[1024, 512, 256, 128, 3])

    print(f"{'='*10} Graph Encoder {'='*10}")
    print(encoder)
    print(f"{'='*10} MLP {'='*10}")
    print(mlp)
    print(f"{'='*10} Decoder {'='*10}")
    print(decoder)

    x = torch.randn(100, 3)
    out = encoder(x)
    print(f'encoder output shape: {out.shape}')
    out = mlp(out)
    print(f'mlp output shape: {out.shape}')
    out = decoder(out)
    print(f'decoder output shape: {out.shape}')
