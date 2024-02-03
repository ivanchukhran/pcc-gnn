import torch
from torch import nn
import torch_geometric


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers 
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.convs.append(self.conv1)
        for i in range(num_layers - 2):
            self.convs.append(torch_geometric.nn.GCNConv(hidden_channels, hidden_channels))
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, out_channels)
        self.convs.append(self.conv2)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
        return x
    

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphDecoder, self).__init__()
        self.num_layers = num_layers 
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.convs.append(self.conv1)
        for i in range(num_layers - 2):
            self.convs.append(torch_geometric.nn.GCNConv(hidden_channels, hidden_channels))
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, out_channels)
        self.convs.append(self.conv2)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
        return x

class GraphPointNet(nn.Module):
    def __init__(self): pass

    def forward(self, x): pass