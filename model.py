import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import knn_graph


def encoder_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2 * in_channels, out_channels), # x2 because of the edge features
        nn.ReLU()
    )

def decoder_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )


class GraphEncoder(nn.Module):
    """
    GraphEncoder for Point Completion Network on 3D Point Clouds

    Parameters:

    :param k: int - number of nearest neighbors to consider in the graph
    """

    def __init__(self, k: int):
        super(GraphEncoder, self).__init__()
        self.k = k
        self.conv0 = EdgeConv(encoder_block(3, 64))
        self.conv1 = EdgeConv(encoder_block(64, 128))
        self.conv2 = EdgeConv(encoder_block(128, 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph encoder

        :param x: torch.tensor - input tensor of shape (num_nodes, in_channels)
        :param edge_index: torch.tensor - edge index tensor of shape (2, num_edges)
        :return: torch.tensor - output tensor of shape (num_nodes, out_channels)
        """
        edge_index = knn_graph(x, self.k)
        x = self.conv0(x, edge_index)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class PCN(nn.Module):
    def __init__(self, k: int):
        super(PCN, self).__init__()
        self.encoder = GraphEncoder(k)
        self.decoder = Decoder(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x



if __name__ == '__main__':
    encoder = GraphEncoder(k=10)
    x = torch.rand((100, 3))
    output = encoder(x)
    print('\nOutput shape:\n')
    print(output.shape)
    print('\n')

    decoder = Decoder()
    output = decoder(output)
    print('\nOutput shape:\n')
    print(output.shape)
    print('\n')
