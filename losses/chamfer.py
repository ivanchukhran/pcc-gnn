#!/usr/bin/env python
import torch
from torch import device, nn

class ChamferDistance(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_cuda = torch.cuda.is_available()

    def forward(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute the Chamfer distance between two point clouds.
        Args:
            ground_truth (torch.Tensor): The ground truth point cloud.
            prediction (torch.Tensor): The predicted point cloud.
        Returns:
            torch.Tensor: The Chamfer distance between the two point clouds.
        """

        P = self.pairwise_distance(ground_truth, prediction)
        mins, _ = torch.min(P, dim=1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, dim=2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bath_size, num_points_x, point_features = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 100, 3).to(device)
    y = torch.rand(1, 100, 3).to(device)
    simple_dist = ChamferDistance()(x, y)
    print(simple_dist)
