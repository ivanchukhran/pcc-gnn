import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, global_max_pool, MaxAggregation


def encoder_block(in_channels, out_channels):
    return nn.Sequential(
        GCNConv(in_channels, out_channels),
        nn.ReLU(),
        MaxAggregation()
    )


class GraphEncoder(nn.Module):
    """
    GraphEncoder for Point Completion Network on 3D Point Clouds

    Parameters:

    :param in_channels: int - number of input channels
    :param hidden_channels: int - number of hidden channels
    :param out_channels: int - number of output channels (latent dimensionality of the graph encoder)
    :param num_layers: int - number of layers in the graph encoder from input to output
    """
    def __init__(self, latent_dim: int):
        super(GraphEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = encoder_block(3, 64)
        self.conv2 = encoder_block(64, 128)
        self.conv3 = encoder_block(128, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph encoder

        :param x: torch.tensor - input tensor of shape (num_nodes, in_channels)
        :param edge_index: torch.tensor - edge index tensor of shape (2, num_edges)
        :return: torch.tensor - output tensor of shape (num_nodes, out_channels)
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
