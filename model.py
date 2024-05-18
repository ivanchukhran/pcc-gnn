import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.nn import knn_graph, global_max_pool

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

    def __init__(self, blocks: list[int], k: int = 20):
        super(GraphEncoder, self).__init__()
        self.k = k
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

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the graph encoder

        Args:
            - x: torch.tensor - input tensor of shape (num_nodes, in_channels)
            - edge_index: torch.tensor - edge index tensor of shape (2, num_edges)
        Return:
            - torch.tensor - output tensor of shape (num_nodes, out_channels)
        """
        edge_index = knn_graph(x, self.k, batch=batch, loop=False)
        for module in self.encoder_modules:
            x = module(x, edge_index)
        return x

    def __repr__(self) -> str:
        return f"GraphEncoder(k={self.k}, blocks={self.blocks})"

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


class GraphPointCompletionNetwork(nn.Module):
    def __init__(self, cfg: dict):
        super(GraphPointCompletionNetwork, self).__init__()
        if not cfg:
            raise ValueError("The configuration dictionary should not be empty.")
        if not (cfg.get('encoder') and cfg.get('mapping_network') and cfg.get('decoder')):
            raise ValueError("The configuration dictionary should contain encoder, mapping_network and decoder.")
        self.num_coarse: int = cfg.get('num_coarse', 1024)
        self.num_dense: int = cfg.get('num_dense', 16384)
        self.encoder = GraphEncoder(**cfg['encoder'])
        self.mapping_network = MappingNetwork(**cfg['mapping_network'], num_coarse=self.num_coarse)
        self.decoder = Decoder(**cfg['decoder'])
        self.grid_size: int = cfg.get('grid_size', 4)
        self.grid_scale: int = cfg.get('grid_scale', 0.05)
        a = (torch.linspace(-self.grid_scale, self.grid_scale, steps=self.grid_size, dtype=torch.float)
             .view(1, self.grid_size)
             .expand(self.grid_size, self.grid_size)
             .reshape(1, -1))
        b = (torch.linspace(-self.grid_scale, self.grid_scale, steps=self.grid_size, dtype=torch.float)
             .view(self.grid_size, 1)
             .expand(self.grid_size, self.grid_size)
             .reshape(1, -1))
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # (1, 2, S)
        if torch.cuda.is_available():
            self.folding_seed = self.folding_seed.cuda()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        if x.dim() == 3:
            x = x.squeeze(0)
        # As the knn_graph method requires a batch tensor, we create a batch tensor of size B
        batch = torch.arange(B, device=x.device).view(-1, 1).repeat(1, N).view(-1)

        # Encoder
        feature = self.encoder(x, batch=batch)                                              # (B * N, 1024)
        feature = feature.unsqueeze(2).reshape(B, -1, N)                                    # (B, N, 1024)
        # feature = feature.view(B, N, -1).transpose(2, 1)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2)[0]                                       # (B, 1024)
        # feature_global = feature_global.unsqueeze(1).expand(-1, N, -1)                      # (B, 1024, N)
        # feature = torch.cat([feature, feature_global.transpose(2, 1)], dim=1)               # (B, 2048, N)

        # Mapping Network
        coarse = self.mapping_network(feature_global).view(B, -1, 3)                        # (B, num_coarse * N, 3)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)            # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_dense)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)            # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                          # (B, 2, num_dense)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_dense)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_dense)

        # Decoder
        fine = self.decoder(feat) + point_feat
        return coarse.contiguous(), fine.transpose(2, 1).contiguous()

if __name__ == '__main__':
    encoder_blocks = [3, 64, 128, 256, 512, 1024]
    decoder_blocks = [1024 + 2 + 3, 512, 256, 128, 64, 3]
    latent_dim = max(encoder_blocks)
    cfg = {
        'encoder': {'blocks': encoder_blocks, 'k': 20},
        'mapping_network': {'latent_dim': latent_dim, 'n_blocks': 3},
        'decoder': {'blocks': decoder_blocks, 'grid_size': 4, 'grid_scale': 0.05},
        'num_dense': 16384,
        'num_coarse': 1024,
    }
    model = GraphPointCompletionNetwork(cfg)
    x = torch.randn(1, 100,  3)
    coarse, fine = model(x)
    print(f'coarse shape: {coarse.shape}')
    print(f'fine shape: {fine.shape}')
