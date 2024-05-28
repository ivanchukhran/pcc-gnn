#!/usr/bin/env python3
import os
import re
import json
from typing import Dict, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from models import GraphPointCompletionNetwork
from datasets import GraphShapeNet
from losses import ChamferDistance
from utils.pc import save_plot
from utils.telelogger import TeleLogger
from setup import setup_logging

logs_dir = os.path.join('logs', 'train')
logger = setup_logging(logs_dir)

def train_step(model: nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: Callable | nn.Module,
               loss_multiplier: float,
               device: torch.device) -> tuple:
    """
    Run the model through the whole train set.

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

def validation_step(model: nn.Module,
                    dataloader: DataLoader,
                    loss_fn: Callable | nn.Module,
                    device: torch.device,
                    loss_multiplier: float) -> tuple:
    """
    Run the model through the whole validation set.

    Arguments:
        model: nn.Module - The model to train.
        dataloader: torch.utils.data.DataLoader - The dataloader to use.
        loss_fn: Callable | nn.Module - The loss function to use.
        device: torch.device - The device to use.

    """
    final_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            if (batches := batch.batch) is not None:
                batch_size = batches.max().item() + 1
            else:
                batch_size = 1

            ground_truth = batch.y.to(device)
            ground_truth = ground_truth.view(batch_size, -1, 3)

            coarse_predicted, dense_predicted = model(batch.to(device))
            num_coarse_points = coarse_predicted.shape[1]
            coarse_predicted = coarse_predicted.view(batch_size, -1, 3)
            dense_predicted = dense_predicted.view(batch_size, -1, 3)
            coarse_loss = loss_fn(coarse_predicted, ground_truth[:, :num_coarse_points, :])
            dense_loss = loss_fn(dense_predicted, ground_truth)
            loss = coarse_loss + loss_multiplier * dense_loss

            final_loss += loss.item()
        final_loss /= len(dataloader)
    return final_loss, (batch.x.reshape(batch_size, -1, 3).detach().cpu(),
                        dense_predicted.detach().cpu(),
                        ground_truth.detach().cpu())

restore_policies = ['latest', 'best_loss']
log_regex = re.compile(r'Epoch: (\d+), Train Loss: (\d+\.\d+), Validation Loss: (\d+\.\d+)(, new best loss)?')

def restore_metrics(policy: str, weight_dir: str | None = None, logs_dir: str | None = None) -> tuple:
    """
    Restore the best loss and epoch from the previous training.

    Arguments:
        policy: str - The policy to use to restore the metrics.
        path: str - The path to the saved model.

    Returns:
        best_loss: float - The best loss.
        best_epoch: int - The best epoch.
    """
    if policy not in restore_policies:
        raise ValueError(f'Invalid policy. Choose from {restore_policies}')
    best_loss = float('inf')
    best_epoch = 0
    # return the default values if dirs does not exist
    if not (weight_dir and os.path.exists(weight_dir)) or not (logs_dir and os.path.exists(logs_dir)):
        return best_loss, best_epoch

    pytorch_weights = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
    epochs = [int(filename.split('_')[-1].split('.')[0]) for filename in pytorch_weights]
    logs_file = os.path.join(logs_dir, 'logs.log')
    if not os.path.exists(logs_file):
        return best_loss, best_epoch
    with open(logs_file, 'r') as file:
        logs = file.readlines()
    match policy:
        case 'latest':
            best_epoch = max(epochs) if epochs else 0
            for log in logs[::-1]:
                if (matches := log_regex.match(log)):
                    epoch, _, val_loss, new_best = matches.groups()
                    epoch = int(epoch)
                    val_loss = float(val_loss)
                    if epoch == best_epoch:
                        best_loss = val_loss
                        break
            if best_loss == float('inf'):
                best_epoch = 0
                logger.warning(f'Could not find the best loss in the logs file. Using the defualt value {best_loss}.')
        case 'best_loss':
            for log in logs[::-1]:
                if (matches := log_regex.match(log)):
                    epoch, _, val_loss, new_best = matches.groups()
                    if new_best:
                        best_loss = float(val_loss)
                        best_epoch = int(epoch)
                        break
            if best_epoch not in epochs:
                best_epoch = 0
                best_loss = float('inf')
    return best_loss, best_epoch

def restore_model_state(epoch: int, model: nn.Module, optimizer: Optimizer, save_path: str) -> None:
    weight_path = os.path.join(save_path, f'model_{epoch}.pt')
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])

def train(model: nn.Module,
          dataloaders: dict[str, DataLoader],
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
    best_loss, best_epoch = restore_metrics(policy='latest', weight_dir=save_path, logs_dir=logs_dir)
    logger.info(f'Restored best loss: {best_loss}, best epoch: {best_epoch}')

    if best_epoch > 0:
        restore_model_state(epoch=best_epoch, model=model, optimizer=optimizer, save_path=save_path)

    train_dataloader = dataloaders['train']
    validation_dataloader = dataloaders['validation']

    for epoch in tqdm(range(epochs), desc='Epochs'):
        model, optimizer, loss, existing, reconstructed, ground_truth = train_step(model, train_dataloader, optimizer, loss_fn, loss_multiplier, device)
        if scheduler:
            scheduler.step()
        validation_loss, validation_samples = validation_step(model, validation_dataloader, loss_fn, device, loss_multiplier)

        log_message = f'Epoch: {epoch + 1}, Train Loss: {loss:.4}, Validation Loss: {validation_loss:.4f}'

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            log_message += f', new best loss'

        logger.info(log_message)

        if (epoch + 1) % save_interval == 0 :
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            full_save_path = os.path.join(save_path, f'model_{epoch + 1}.pt')
            torch.save(state_dict, full_save_path)
            if telelogger:
                save_sample_path = os.path.join(save_path, save_sample_path)

                plots_to_send: list[str] = []
                plots_to_send.append(save_plot(existing[0].numpy(), epoch=epoch, save_path=save_sample_path, type_='existing'))
                plots_to_send.append(save_plot(reconstructed[1].numpy(), epoch=epoch, save_path=save_sample_path, type_='reconstructed'))
                plots_to_send.append(save_plot(ground_truth[2].numpy(), epoch=epoch, save_path=save_sample_path, type_='ground_truth'))

                validiation_plots_to_send: list[str] = []
                validiation_plots_to_send.append(save_plot(validation_samples[0].numpy(), epoch=epoch, save_path=save_sample_path, type_='val_existing'))
                validiation_plots_to_send.append(save_plot(validation_samples[1].numpy(), epoch=epoch, save_path=save_sample_path, type_='val_reconstructed'))
                validiation_plots_to_send.append(save_plot(validation_samples[2].numpy(), epoch=epoch, save_path=save_sample_path, type_='val_ground_truth'))
                plots_to_send.extend(validiation_plots_to_send)
                telelogger.send_message_with_media(media=plots_to_send, message=log_message) # Send only the first 9 plots because of Telegram's limit

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
    train_dataset = GraphShapeNet(**dataset_config.get('train'))
    validation_dataset = GraphShapeNet(**dataset_config.get('validation'))
    # just for testing
    # data = Data(x=torch.rand(1024, 3), edge_index=torch.randint(0, 1024, (2, 1024)), y=torch.rand(2048, 3))
    # train_dataset = [data for _ in range(8)]
    # validation_dataset = [data for _ in range(8)]
    dataloader_config = config.get('dataloader')
    if not dataloader_config:
        raise ValueError('Dataloader configuration not found in config file or empty.')
    dataloaders = {
        'train': DataLoader(train_dataset, **dataloader_config),
        'validation': DataLoader(validation_dataset, **dataloader_config)
    }

    epochs = config.get('epochs', 100)
    # epochs = 1
    save_path = config.get('save_path', 'checkpoints')
    save_interval = config.get('save_interval', 100)
    # save_interval = 1
    telelogger = None
    if params.telelogger:
        with open(params.telelogger, 'r') as file:
            telelogger_config = json.load(file)
        telelogger = TeleLogger(**telelogger_config)

    save_samples_path = os.path.join(save_path, 'samples')
    dirs_to_create = [save_path, save_samples_path]
    for dir_ in dirs_to_create:
        os.makedirs(dir_, exist_ok=True)

    train(model, dataloaders, optimizer, loss_fn, loss_multiplier, device, epochs,
          save_path, save_interval, scheduler=scheduler, telelogger=telelogger)


if __name__ == '__main__':
    main()
