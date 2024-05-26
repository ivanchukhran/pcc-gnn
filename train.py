import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training script')


def train(args):
    cfg_path = args.cfg
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Config file not found at {cfg_path}')
    with open(args.cfg, 'r') as f:
        config = json.load(f)
    # config_path = args.cfg
    # if not os.path.exists(config_path):
    #     raise FileNotFoundError(f'Config file not found at {config_path}')

    # with open(config_path, 'r') as f:
    #     config = json.load(f)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = PCN(k=config.get('k')).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr'))
    # # TODO: Implement the ChamferLoss
    # chamfer_loss = ChamferLoss()

    # # TODO: Implement the ShapeNetDataset
    # dataset = ShapeNetDataset(root=args.root, split='train')
    # dataloader = DataLoader(dataset, batch_size=config.get('batch_size'), shuffle=True)

    # model.train()

    # for epoch in range(args.epochs):
    #     for i, data in enumerate(dataloader):
    #         data = data.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = chamfer_loss(data, output)
    #         loss.backward()
    #         optimizer.step()
    #         # TODO: reimplement the print statement with the custom logger
    #         # TODO: change the logging frequency wrt the config file
    #         if i % 100 == 0:
    #             print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')


def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()
