import os
import sys

import pytest
import torch

from core.train import restore_metrics

weights_base_path = os.path.join('tmp', 'weights')
logs_base_path = os.path.join('tmp', 'logs')

DEFAULT_EPOCH = 0
DEFAULT_LOSS = float('inf')

latest = 'latest'
best_loss = 'best_loss'

###########################################################
### Test cases for restore_metrics with 'latest' policy ###
###########################################################
def test_latest_restore_metrics_with_no_files_should_return_default_metrics():
    loss, epoch = restore_metrics(policy=latest, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'

def test_latest_restore_metrics_with_weights_file_should_return_default_metrics():
    os.makedirs(weights_base_path, exist_ok=True)
    torch.save({'epoch': 1, 'loss': 0.1}, os.path.join(weights_base_path, 'model_1.pth'))
    loss, epoch = restore_metrics(policy=latest, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(weights_base_path, 'model_1.pt'))
    except FileNotFoundError:
        pass

def test_latest_restore_metrics_with_logs_file_should_return_default_metrics():
    os.makedirs(logs_base_path, exist_ok=True)
    log_string = f'Epoch: 1, Train Loss: 0.1000, Val Loss: 0.2000'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=latest, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass

def test_latest_restore_metrics_with_logs_file_without_epoch_should_return_default_metrics():
    os.makedirs(logs_base_path, exist_ok=True)
    log_string = f'blah blah blah'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=latest, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass

def test_latest_restore_metrics_with_weights_and_logs_files_with_same_progress_should_return_latest_common_metrics():
    os.makedirs(weights_base_path, exist_ok=True)
    os.makedirs(logs_base_path, exist_ok=True)
    torch.save({'epoch': 1, 'loss': 0.1}, os.path.join(weights_base_path, 'model_1.pt'))
    log_string = f'Epoch: 1, Train Loss: 0.1000, Validation Loss: 0.2000'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=latest, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == 1, f'Expected 1, but got {epoch}'
    assert loss == 0.2000, f'Expected 0.2000, but got {loss}'
    try:
        os.remove(os.path.join(weights_base_path, 'model_1.pth'))
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass

def test_latest_restore_metrics_with_weights_and_logs_files_with_different_progress_should_return_default_metrics():
    os.makedirs(weights_base_path, exist_ok=True)
    os.makedirs(logs_base_path, exist_ok=True)
    torch.save({'epoch': 1, 'loss': 0.1}, os.path.join(weights_base_path, 'model_1.pth'))
    log_string = f'Epoch: 1, Train Loss: 0.1000, Val Loss: 0.2000\nEpoch: 2, Train Loss: 0.1000, Val Loss: 0.2000'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=latest, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(weights_base_path, 'model_1.pth'))
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass

##############################################################
### Test cases for restore_metrics with 'best_loss' policy ###
##############################################################
def test_best_loss_restore_metrics_with_no_files_should_return_default_metrics():
    loss, epoch = restore_metrics(policy=best_loss, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'

def test_best_loss_restore_metrics_with_weights_file_should_return_default_metrics():
    os.makedirs(weights_base_path, exist_ok=True)
    torch.save({'epoch': 1, 'loss': 0.1}, os.path.join(weights_base_path, 'model_1.pth'))
    loss, epoch = restore_metrics(policy=best_loss, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(weights_base_path, 'model_1.pt'))
    except FileNotFoundError:
        pass

def test_best_loss_restore_metrics_with_logs_file_should_return_default_metrics():
    os.makedirs(logs_base_path, exist_ok=True)
    log_string = f'Epoch: 1, Train Loss: 0.1000, Val Loss: 0.2000'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=best_loss, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass

def test_best_loss_restore_metrics_with_weights_and_logs_files_with_same_progress_should_return_best_loss_metrics():
    os.makedirs(weights_base_path, exist_ok=True)
    os.makedirs(logs_base_path, exist_ok=True)
    torch.save({'epoch': 1, 'loss': 0.1}, os.path.join(weights_base_path, 'model_1.pt'))
    log_string = f'Epoch: 1, Train Loss: 0.1000, Validation Loss: 0.2000, new best loss'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=best_loss, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == 1, f'Expected 1, but got {epoch}'
    assert loss == 0.2000, f'Expected 0.2000, but got {loss}'
    try:
        os.remove(os.path.join(weights_base_path, 'model_1.pth'))
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass

def test_best_loss_restore_metrics_with_weights_and_logs_files_with_different_progress_should_return_best_loss_metrics():
    os.makedirs(weights_base_path, exist_ok=True)
    os.makedirs(logs_base_path, exist_ok=True)
    torch.save({'epoch': 1, 'loss': 0.1}, os.path.join(weights_base_path, 'model_1.pt'))
    log_string = f'Epoch: 1, Train Loss: 0.1000, Val Loss: 0.2000\nEpoch: 2, Train Loss: 0.1000, Val Loss: 0.2000, new best loss'
    with open(os.path.join(logs_base_path, 'logs.log'), 'w') as f:
        f.write(log_string)
    loss, epoch = restore_metrics(policy=best_loss, weight_dir=weights_base_path, logs_dir=logs_base_path)
    assert epoch == DEFAULT_EPOCH, f'Expected {DEFAULT_EPOCH}, but got {epoch}'
    assert loss == DEFAULT_LOSS, f'Expected {DEFAULT_LOSS}, but got {loss}'
    try:
        os.remove(os.path.join(weights_base_path, 'model_1.pth'))
        os.remove(os.path.join(logs_base_path, 'logs.log'))
    except FileNotFoundError:
        pass
