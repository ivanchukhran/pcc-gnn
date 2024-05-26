#!/usr/bin/env python3
import os
import json
from typing import Dict, Callable

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from models import GraphPointCompletionNetwork
from datasets import GraphShapeNet
from losses import ChamferDistance
from utils.pc import save_plot
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
          telelogger: TeleLogger | None = None,
          save_sample_path: str = 'samples') -> None:
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
        model, optimizer, loss, existing, reconstructed, ground_truth = train_step(model, dataloader, optimizer, loss_fn, loss_multiplier, device)
        log_message = f'Epoch: {epoch + 1}, Loss: {loss:.4f}'
        logger.info(log_message)
        if scheduler:
            scheduler.step()
        if (epoch + 1) % save_interval == 0:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            full_save_path = os.path.join(save_path, f'model_{epoch + 1}.pt')
            torch.save(state_dict, full_save_path)
            if telelogger:
                plots_to_send = []
                save_sample_path = os.path.join(save_path, save_sample_path)
                os.makedirs(save_sample_path, exist_ok=True)
                for i in range(existing.shape[0]):
                    plots_to_send.append(save_plot(existing[i].numpy(), epoch=epoch, save_path=save_sample_path, type_='existing'))
                    plots_to_send.append(save_plot(reconstructed[i].numpy(), epoch=epoch, save_path=save_sample_path, type_='reconstructed'))
                    plots_to_send.append(save_plot(ground_truth[i].numpy(), epoch=epoch, save_path=save_sample_path, type_='ground_truth'))
                telelogger.send_message_with_media(media=plots_to_send[:9], message=log_message) # Send only the first 9 plots because of Telegram's limit

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
    model = GraphPointCompletionNetwork(model_config).to(device)
    optimizer = getattr(torch.optim, optimizer_config.get('type', 'Adam'))
    optimizer = optimizer(model.parameters(), **optimizer_config.get('hyperparameters', {}))
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
    # just for testing
    # data = Data(x=torch.rand(1024, 3), edge_index=torch.randint(0, 1024, (2, 1024)), y=torch.rand(2048, 3))
    # dataset = [data for _ in range(8)]
    dataloader_config = config.get('dataloader')
    if not dataloader_config:
        raise ValueError('Dataloader configuration not found in config file or empty.')
    dataloader = DataLoader(dataset, **dataloader_config)
    epochs = config.get('epochs', 100)
    # epochs = 1
    save_path = config.get('save_path', 'checkpoints')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_interval = config.get('save_interval', 100)
    # save_interval = 1
    telelogger = None
    if params.telelogger:
        with open(params.telelogger, 'r') as file:
            telelogger_config = json.load(file)
        telelogger = TeleLogger(**telelogger_config)
    train(model, dataloader, optimizer, loss_fn, loss_multiplier, device, epochs,
          save_path, save_interval, scheduler=scheduler, telelogger=telelogger)


if __name__ == '__main__':
    main()
