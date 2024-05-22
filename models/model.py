import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool

from encoder import GraphEncoder
from mapping_network import MappingNetwork
from decoder import Decoder

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
        self.grid_size: int = cfg.get('grid_size', 4)
        self.grid_scale: int = cfg.get('grid_scale', 0.05)
        self.decoder = Decoder(**cfg['decoder'], grid_size=self.grid_size, grid_scale=self.grid_scale)
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

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        B = 1 if data.batch is None else int(data.batch.max().item() + 1)
        x = data.x
        if x is None:
            raise ValueError("The data object should contain the x attribute.")
        N = data.num_nodes if data.num_nodes is not None else x.shape[0]
        N = N // B  # all point clouds have the same number of points

        feature_global = self.encoder(data.x, data.edge_index, data.batch)                                          # (B, 1024)
        feature_global = feature_global.unsqueeze(1).expand(B, N, -1)                                               # (B, N, 1024)
        # Mapping Network
        coarse = self.mapping_network(feature_global).view(B, -1, 3)                                                # (B, num_coarse * N, 3)
        point_feat = coarse.unsqueeze(2).expand(B,-1, self.grid_size ** 2, 3)                                       # (B, num_coarse * N, S, 3)
        point_feat = point_feat.reshape(-1, 3, self.num_dense)                                                      # (B * N, 3, num_dense)
        seed = self.folding_seed.unsqueeze(2).expand(B * N, -1, self.num_coarse, -1)                                # (B, 2, num_coarse, S)
        seed = seed.reshape(B * N, -1, self.num_dense)                                                              # (B, 2, num_dense)

        feature_global = feature_global.reshape(-1, self.num_coarse).unsqueeze(2).expand(-1, -1, self.num_dense)    # (B * N, num_coarse)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                                                 # (B, 1024+2+3, num_dense)

        # Decoder
        fine = self.decoder(feat) + point_feat
        return coarse.reshape(B, N, -1, 3).contiguous(), fine.transpose(2, 1).reshape(B, N, -1, 3).contiguous()

if __name__ == '__main__':
    encoder_blocks = [3, 64, 128, 256, 512, 1024]
    decoder_blocks = [1024 + 2 + 3, 512, 256, 128, 64, 3]
    latent_dim = max(encoder_blocks)
    cfg = {
        'encoder': {'blocks': encoder_blocks},
        'mapping_network': {'latent_dim': latent_dim, 'n_blocks': 3},
        'decoder': {'blocks': decoder_blocks},
        'grid_size': 2,
        'grid_scale': 0.05,
        'num_dense': 4096,
        'num_coarse': 1024,
    }
    model = GraphPointCompletionNetwork(cfg)
    x = torch.randn(100,  3)
    edge_index = torch.randint(0, 100, (2, 100))
    data = Data(x=x, edge_index=edge_index)
    data_list = [data, data]
    dataloader = DataLoader(data_list, batch_size=2, shuffle=True)
    coarse, fine = model(next(iter(dataloader)))
    print(f'coarse shape: {coarse.shape}')
    print(f'fine shape: {fine.shape}')
