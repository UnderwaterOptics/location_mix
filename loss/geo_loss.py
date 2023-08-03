import pdb
import numpy as np

import torch
from torch import nn

from data.geometry import Geometry, Vector3D


class GeoLoss(nn.Module):
    def __init__(self, r) -> None:
        super().__init__()

        self.register_buffer('r', torch.tensor(r, dtype=torch.float32))
        self.register_buffer('pi', torch.tensor(np.pi, dtype=torch.float32))
        self.register_buffer('sqrt3', torch.tensor(np.sqrt(3), dtype=torch.float32))

        # A = torch.tensor([-r / 2., 0, -self.sqrt3 * r / 6.], dtype=torch.float32)
        # B = torch.tensor([ r / 2., 0, -self.sqrt3 * r / 6.], dtype=torch.float32)
        # C = torch.tensor([      0, 0,  self.sqrt3 * r / 3.], dtype=torch.float32)
        A = torch.tensor([0.,       0.,  self.sqrt3 * r / 3.], dtype=torch.float32)
        B = torch.tensor([0., - r / 2., -self.sqrt3 * r / 6.], dtype=torch.float32)
        C = torch.tensor([0.,   r / 2., -self.sqrt3 * r / 6.], dtype=torch.float32)
        self.register_buffer('A', A)
        self.register_buffer('B', B)
        self.register_buffer('C', C)
    
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, cord_p, batch, lambda_axis, train):

        self.loss = self.mse if train else self.mae
        inc_angle_p, d_p = self.cord2rad(cord_p)
        # different lambda_axis for train and test
        # pdb.set_trace()
        if train:
            weighted_cord_loss = torch.sum(
            lambda_axis[0] * (cord_p[:, 0] - batch['cord'][:, 0])**2 + \
            lambda_axis[1] * (cord_p[:, 1] - batch['cord'][:, 1])**2 + \
            lambda_axis[2] * (cord_p[:, 2] - batch['cord'][:, 2])**2) / cord_p.shape[0] / (lambda_axis[0] + lambda_axis[1] + lambda_axis[2])
        else:
            weighted_cord_loss = torch.sum(
            lambda_axis[0] * torch.abs(cord_p[:, 0] - batch['cord'][:, 0]) + \
            lambda_axis[1] * torch.abs(cord_p[:, 1] - batch['cord'][:, 1]) + \
            lambda_axis[2] * torch.abs(cord_p[:, 2] - batch['cord'][:, 2])) / cord_p.shape[0] / (lambda_axis[0] + lambda_axis[1] + lambda_axis[2])
        cord_loss = self.loss(cord_p, batch['cord'])    
        
        # rad loss or ang loss
        inc_angle = batch['inc_angle']['inc_angle']
        inc_angle_loss = self.loss(inc_angle_p, inc_angle)
        # distance loss 
        d_loss = self.loss(d_p, batch['d'])
        return weighted_cord_loss, cord_loss, inc_angle_loss, d_loss

    def cord2rad(self, cord):
        # pdb.set_trace()
        batch_size = cord.shape[0]
        A = self.A.unsqueeze(0).repeat((batch_size, 1))
        B = self.B.unsqueeze(0).repeat((batch_size, 1))
        C = self.C.unsqueeze(0).repeat((batch_size, 1))
        # pdb.set_trace()
        OA = A - cord
        OB = B - cord
        OC = C - cord

        alpha = self.arccos(OA, OB)
        beta = self.arccos(OB, OC)
        gamma = self.arccos(OC, OA)

        d = torch.norm(cord, 2, dim=1)
        # pdb.set_trace()
        return torch.stack([alpha, beta, gamma], dim=1), d    

    def arccos(self, OA, OB):
        return torch.arccos(torch.einsum('ij, ij -> i', [OA, OB]) / (torch.norm(OA, 2, dim=1) * torch.norm(OB, 2, dim=1)))

    def rad2deg(self, rad):
        return rad * 180 / self.pi

    def deg2rad(self, deg):
        return deg * self.pi / 180.



        


