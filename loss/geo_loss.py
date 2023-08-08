import pdb
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


from data.geometry import Vector3D


class GeoLoss(nn.Module):
    def __init__(self, r, data_expansion) -> None:
        super().__init__()
        self.register_buffer('data_expansion', torch.tensor(data_expansion, dtype=torch.float32))
        self.register_buffer('r', torch.tensor(r * data_expansion, dtype=torch.float32))
        self.register_buffer('pi', torch.tensor(np.pi, dtype=torch.float32))
        self.register_buffer('sqrt3', torch.tensor(np.sqrt(3), dtype=torch.float32))

        A = Vector3D([0.,       0.,  self.sqrt3 * self.r / 3.]) 
        B = Vector3D([0., - self.r / 2., -self.sqrt3 * self.r / 6.]) 
        C = Vector3D([0.,   self.r / 2., -self.sqrt3 * self.r / 6.]) 

        self.register_buffer('A', A)
        self.register_buffer('B', B)
        self.register_buffer('C', C)
    
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, a1, b1, c1, A1_d, B1_d, C1_d, t, d, 
                t_p, d_p):
        # t_p (b, 3)
        # d_p (b, 1)
        batch_size = t.shape[0]
        A = self.A.unsqueeze(0).repeat((batch_size, 1))
        B = self.B.unsqueeze(0).repeat((batch_size, 1))
        C = self.C.unsqueeze(0).repeat((batch_size, 1)) 

        # 灯向量
        T_p = t_p * d_p
        A1_d_p, a1_p = self.cord2vec(A, T_p)
        B1_d_p, b1_p = self.cord2vec(B, T_p)
        C1_d_p, c1_p = self.cord2vec(C, T_p)

        # 方向的一致性
        # 向量都计算余弦损失
        # 平移向量
        t_closs = self.cosine_loss(t, t_p).mean()

        a1_closs = self.cosine_loss(a1_p, a1).mean()
        b1_closs = self.cosine_loss(b1_p, b1).mean()
        c1_closs = self.cosine_loss(c1_p, c1).mean()

        abc_closs = (a1_closs + b1_closs + c1_closs) / 3.

        # 距离的一致性
        # 将距离单位转换为mm
        T_dloss = self.mae(d_p, d) / self.data_expansion * 10
        A1_dloss = self.mae(A1_d_p, A1_d.unsqueeze(dim=1))
        B1_dloss = self.mae(B1_d_p, B1_d.unsqueeze(dim=1))
        C1_dloss = self.mae(C1_d_p, C1_d.unsqueeze(dim=1))

        ABC_dloss =  (A1_dloss + B1_dloss + C1_dloss) / self.data_expansion * 10 / 3.
        # pdb.set_trace()

        return t_closs, abc_closs, T_dloss, ABC_dloss
    
    def cosine_loss(self, A, B, dim = 1):
        return 1 - F.cosine_similarity(A, B, dim)
    
    def cord2vec(self, cord, T):
        # pdb.set_trace()
        cord1 = cord.add(T)
        d = cord1.norm(dim=1).unsqueeze(dim=1)
        vec = cord1 / d
        return d, vec
        










    # def forward(self, cord_p, batch, lambda_axis, train):

    #     self.loss = self.mse if train else self.mae
    #     inc_angle_p, d_p = self.cord2rad(cord_p)
    #     # different lambda_axis for train and test
    #     # pdb.set_trace()
    #     if train:
    #         weighted_cord_loss = torch.sum(
    #         lambda_axis[0] * (cord_p[:, 0] - batch['cord'][:, 0])**2 + \
    #         lambda_axis[1] * (cord_p[:, 1] - batch['cord'][:, 1])**2 + \
    #         lambda_axis[2] * (cord_p[:, 2] - batch['cord'][:, 2])**2) / cord_p.shape[0] / (lambda_axis[0] + lambda_axis[1] + lambda_axis[2])
    #     else:
    #         weighted_cord_loss = torch.sum(
    #         lambda_axis[0] * torch.abs(cord_p[:, 0] - batch['cord'][:, 0]) + \
    #         lambda_axis[1] * torch.abs(cord_p[:, 1] - batch['cord'][:, 1]) + \
    #         lambda_axis[2] * torch.abs(cord_p[:, 2] - batch['cord'][:, 2])) / cord_p.shape[0] / (lambda_axis[0] + lambda_axis[1] + lambda_axis[2])
    #     cord_loss = self.loss(cord_p, batch['cord'])    
        
    #     # rad loss or ang loss
    #     inc_angle = batch['inc_angle']['inc_angle']
    #     inc_angle_loss = self.loss(inc_angle_p, inc_angle)
    #     # distance loss 
    #     d_loss = self.loss(d_p, batch['d'])
    #     return weighted_cord_loss, cord_loss, inc_angle_loss, d_loss

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



        


