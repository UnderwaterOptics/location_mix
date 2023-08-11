import torch
from pytorch_lightning import LightningDataModule
from torch.distributions import uniform
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pdb

from .geometry import Vector3D

class LocData(Dataset):
    def __init__(self, r, max_rad, max_d, max_x, min_x, data_len) -> None:
        super().__init__()
        self.pi = np.pi
        self.sqrt3 = np.sqrt(3).astype(np.float32)

        self.A = Vector3D([0.,       0.,  self.sqrt3 * r / 3.]) 
        self.B = Vector3D([0., - r / 2., -self.sqrt3 * r / 6.]) 
        self.C = Vector3D([0.,   r / 2., -self.sqrt3 * r / 6.])

        self.max_rad = max_rad
        self.max_d = max_d
        self.max_x = max_x
        # self.min_x = r if min_x == 0. else min_x
        self.len = data_len

        if min_x is None:
            self.min_x = r
        else:
            self.min_x = min_x


    def _sample_data(self):
        '''
        计算数据集的取值范围，并随机均匀采样一个坐标数据，计算对应的方向向量、单位向量和偏角,
        并返回网络的输入, 方向向量. 网络的输出, 平移向量
        '''

        # 采样范围
        x = np.random.uniform(self.min_x, self.max_x)
        yz = np.abs(x, dtype=np.float32) * np.tan(self.max_rad, dtype=np.float32)
        y = np.random.uniform(-yz, yz)
        z = np.random.uniform(-yz, yz)

        # 定义坐标正向平移向量
        T = Vector3D([x, y, z])
        # 将平移向量标准化
        d = T.norm() 
        t = T/d
        # d_n = d / self.max_d

        # 分别计算三个导引灯在AUV坐标系中的位置
        A1, A1_d, a1 = self.cord2vec(self.A, T)
        B1, B1_d, b1 = self.cord2vec(self.B, T)
        C1, C1_d, c1 = self.cord2vec(self.C, T)

        # 计算向量偏角
        A1_h, A1_v = A1.vec2ang
        B1_h, B1_v = B1.vec2ang
        C1_h, C1_v = C1.vec2ang
        
        # 拼接成列向量
        ang = torch.stack([A1_h, A1_v, B1_h, B1_v, C1_h, C1_v])

        # 方向向量 训练模式
        # return a1, b1, c1, A1, B1, C1, T, t, d, self.max_d
        return a1, b1, c1, A1_d, B1_d, C1_d, T, t, d, self.max_d
    

    def cord2vec(self, cord, T):
        cord1 = cord.add(T)
        d = cord1.norm()
        vec = cord1 / d
        return cord1, d, vec
        

    def __getitem__(self, index):
        # 对于采样出来的数据，index实际上是没用的
        return self._sample_data()

    def __len__(self):
        return self.len

class LocDataModule(LightningDataModule):
    def __init__(self, 
            batch_size: int = 4, 
            num_workers: int = 56,
            train_len: int = 10000, 
            data_expansion: int = 10000, 
            r: float = 0.8,
            max_deg: float = 30., 
            max_x: float = 20., 
            vali_max_x: float = 3.,
            vali_min_x: float = 2.,
            ):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_len = train_len
        self.test_len = int(train_len * 0.2)
        self.val_len = int(train_len * 0.2)

        self.r = r * data_expansion
        self.max_x = max_x * data_expansion
        self.vali_max_x = vali_max_x * data_expansion
        self.vali_min_x = vali_min_x * data_expansion

        self.max_rad = np.radians(max_deg, dtype=np.float32)
        self.max_d = np.sqrt(self.max_x * self.max_x + 2 * (self.max_x * np.tan(self.max_rad)) ** 2)

        # self.min_x = min_x
        # self.loc_train = LocData(data_config)
    
    def setup(self, stage=None):
        # self.loc_train = LocData('train', self.r, self.max_deg, self.max_x, self.min_x, self.train_len)
        # self.loc_test  = LocData('test', self.r, self.max_deg, self.max_x, self.min_x, self.test_len)
        # self.loc_val   = LocData('val', self.r, self.max_deg, self.max_x, self.min_x, self.val_len)
        self.loc_train = LocData(self.r, self.max_rad, self.max_d, self.max_x, None, self.train_len)
        self.loc_test  = LocData(self.r, self.max_rad, self.max_d, self.max_x, None, self.test_len)
        self.loc_val   = LocData(self.r, self.max_rad, self.max_d, self.vali_max_x, self.vali_min_x, self.val_len)

    def train_dataloader(self):
        return DataLoader(self.loc_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.loc_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.loc_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.loc_test, batch_size=self.batch_size, num_workers=self.num_workers)
