#!/usr/bin/env python3
from typing import Dict, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import ChamferDistance
from setup import setup_logging

def train_step(model: nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: Callable | nn.Module,
               loss_multiplier: float,
               device: torch.device):
    """
    Train the model for one epoch.

    Arguments:
        model: nn.Module - The model to train.
        dataloader: torch.utils.data.DataLoader - The dataloader to use.
        optimizer: torch.optim.Optimizer - The optimizer to use.
        loss_fn: Callable | nn.Module - The loss function to use.
        loss_multiplier: float - The multiplier to apply to the dense loss.
        device: torch.device - The device to use.
    """
    final_loss = 0.0
    num_batches = len(dataloader)
    model.train()
    for i, data in tqdm(enumerate(dataloader), total=num_batches, desc='Batches', leave=False):
        if (batches := data.batch) is not None:
            batch_size = batches.max().item() + 1
        else:
            batch_size = 1

        optimizer.zero_grad()

        ground_truth = data.y.to(device)
        ground_truth = ground_truth.view(batch_size, -1, 3)

        coarse_predicted, dense_predicted = model(data)
        num_coarse_points = coarse_predicted.shape[1]
        coarse_predicted = coarse_predicted.view(batch_size, -1, 3)
        dense_predicted = dense_predicted.view(batch_size, -1, 3)
        coarse_loss = loss_fn(coarse_predicted, ground_truth[:, :num_coarse_points, :])
        dense_loss = loss_fn(dense_predicted, ground_truth)
        loss = coarse_loss + loss_multiplier * dense_loss

        loss.backward()
        optimizer.step()

        final_loss += loss.item()
    final_loss /= num_batches
    return model, optimizer, final_loss

def train(model: nn.Module,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable | nn.Module,
          loss_multiplier: float,
          device: torch.device,
          epochs: int,
          save_path: str,
          save_interval: int = 1):
    """
    Train the model for multiple epochs.

    Arguments:
        model: nn.Module - The model to train.
        dataloader: torch.utils.data.DataLoader - The dataloader to use.
        optimizer: torch.optim.Optimizer - The optimizer to use.
        loss_fn: Callable | nn.Module - The loss function to use.
        loss_multiplier: float - The multiplier to apply to the dense loss.
        device: torch.device - The device to use.
        epochs: int - The number of epochs to train for.
        save_path: str - The path to save the model to.
        save_interval: int - The interval at which to save the model.
    """
    for epoch in tqdm(range(epochs), desc='Epochs'):
        model, optimizer, loss = train_step(model, dataloader, optimizer, loss_fn, loss_multiplier, device)
        print(f'Epoch: {epoch + 1}, Loss: {loss}')

        if (epoch + 1) % save_interval == 0:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dict, save_path)



if __name__ == '__main__':
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader
    from models import GraphPointCompletionNetwork
    from datasets import GraphShapeNet

    dataset = GraphShapeNet(dir_path='/home/chukhran/datasets/completion/shapenet/ShapeNetPointCloud', split='train')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'Running on device: {device}')

    encoder_blocks = [3, 64, 128, 256, 512]
    decoder_blocks = [512 + 2 + 3, 256, 128, 64, 3]
    latent_dim = max(encoder_blocks)
    cfg = {
        'encoder': {'blocks': encoder_blocks},
        'mapping_network': {'latent_dim': latent_dim, 'n_blocks': 2},
        'decoder': {'blocks': decoder_blocks},
        'grid_size': 2,
        'grid_scale': 0.05,
        'num_dense': 2048,
        'num_coarse': 512,
    }
    model = GraphPointCompletionNetwork(cfg).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = ChamferDistance()
    loss_multiplier = 1.0
    model, optimizer, loss = train_step(model, dataloader, optimizer, loss_fn=loss_fn, loss_multiplier=loss_multiplier, device=device)
    print(f'Loss: {loss}')
