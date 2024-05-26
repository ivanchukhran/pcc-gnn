#!/usr/bin/env python3
import json
from typing import Dict, Callable

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from models import GraphPointCompletionNetwork
from datasets import GraphShapeNet
from losses import ChamferDistance
from utils.telelogger import TeleLogger
from setup import setup_logging

logger = setup_logging('logs/train/')

def train_step(model: nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: Callable | nn.Module,
               loss_multiplier: float,
               device: torch.device) -> tuple:
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

        coarse_predicted, dense_predicted = model(data.to(device))
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
    return (model,                                                      # model
            optimizer,                                                  # optimizer
            final_loss,                                                 # loss
            data.x.reshape(batch_size, -1, 3).detach().cpu(),           # existing points
            dense_predicted.detach().cpu(),                             # predicted points
            ground_truth.reshape(batch_size, -1, 3).detach().cpu())     # ground truth points

def train(model: nn.Module,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable | nn.Module,
          loss_multiplier: float,
          device: torch.device,
          epochs: int,
          save_path: str,
          save_interval: int = 1,
          scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
          telelogger: TeleLogger | None = None):
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
        log_message = f'Epoch: {epoch + 1}, Loss: {loss}'
        logger.debug(log_message)
        model, optimizer, loss, existing, reconstructed, ground_truth = train_step(model, dataloader, optimizer, loss_fn, loss_multiplier, device)
        if scheduler:
            scheduler.step()
        if (epoch + 1) % save_interval == 0:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_path = f'{save_path}/model_{epoch + 1}.pt'
            torch.save(state_dict, save_path)
            # if telelogger:
            #     plots_to_send = []
            #     for i in range(existing.shape[0]):
            #         plots_to_send.append(save_plot(existing[i], epoch=epoch, save_path=save_sample_path, type_='existing'))
            #         plots_to_send.append(save_plot(reconstructed[i], epoch=epoch, save_path=save_sample_path, type_='reconstructed'))
            #         plots_to_send.append(save_plot(ground_truth[i], epoch=epoch, save_path=save_sample_path, type_='ground_truth'))
            #     telelogger.send_message_with_media(media=plots_to_send, message=log_message)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--telelogger', type=str, help='Path to the telelogger configuration file.')
    return parser.parse_args()

def main():
    params = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open (params.config, 'r') as file:
        config = json.load(file)
    model_config = config.get('model')
    optimizer_config = config.get('optimizer')
    if not model_config:
         raise ValueError('Model configuration not found in config file or empty.')
    if not optimizer_config:
        raise ValueError('Optimizer configuration not found in config file or empty.')
    model = GraphPointCompletionNetwork(**model_config).to(device)
    optimizer = getattr(torch.optim, optimizer_config.get('type', 'Adam'))(model.parameters(), **optimizer_config.get('hyperparameters', {}))
    # optimizer = Adam(model.parameters(), **optimizer_config)
    scheduler = None
    scheduler_config = config.get('scheduler')
    if scheduler_config:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
    loss_fn = ChamferDistance()
    loss_multiplier = config.get('loss_multiplier', 1.0)
    dataset_config = config.get('dataset')
    if not dataset_config:
        raise ValueError('Dataset configuration not found in config file or empty.')
    dataset = GraphShapeNet(**dataset_config)
    dataloader_config = config.get('dataloader')
    if not dataloader_config:
        raise ValueError('Dataloader configuration not found in config file or empty.')
    dataloader = DataLoader(dataset, **dataloader_config)
    epochs = config.get('epochs', 100)
    save_path = config.get('save_path', 'checkpoints')
    save_interval = config.get('save_interval', 100)
    train(model, dataloader, optimizer, loss_fn, loss_multiplier, device, epochs, save_path, save_interval, scheduler=scheduler)


if __name__ == '__main__':
    dataset = GraphShapeNet(dir_path='/home/chukhran/datasets/completion/shapenet/ShapeNetPointCloud',
                            split='train',
                            classes=['02691156'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on device: {device}')

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
