import torch
from pytorch_lightning import LightningDataModule
from torch.distributions import uniform
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pdb

from .geometry import Geometry, Vector3D

class PoseData(Dataset, Geometry):
    def __init__(self, r, max_deg, max_x, min_x, data_len) -> None:
        super().__init__(r)
        self.r = r
        self.max_deg = max_deg
        self.max_x = max_x
        self.min_x = min_x
        self.len = data_len
    
    def _sample_data(self, seed):
        max_rad = np.degrees(self.max_deg, dtype =np.float32)
        self.max_hw = (torch.abs(torch.tensor(self.max_x)) * torch.sin(max_rad))[0]

        x = uniform.Uniform(self.min_x, self.max_x).sample([1])
        max_hw = (torch.abs(x) * torch.sin(max_rad))[0]
        w = uniform.Uniform(-max_hw, max_hw).sample([1])
        h = uniform.Uniform(-max_hw, max_hw).sample([1])

class LocData(Dataset, Geometry):
    def __init__(self, r, max_deg, max_x, min_x, data_len) -> None:
        super().__init__(r)
        self.r = r
        self.max_deg = np.float32(max_deg)
        self.max_x = np.float32(max_x)
        # self.min_x = r if min_x == 0. else min_x
        self.len = data_len

        # 根据视场角可以解算出大概最小的x的取值，可能为np.sqrt(6) / 3. *self.r。我们在实际计算中还是使用r作为x最小值
        # self.min_x = np.sqrt(6) /3. *self.r
        if min_x is None:
            self.min_x = np.float32(self.r)
        else:
            self.min_x = np.float32(min_x)
        # 最大弧度，及对应的最大yOz取值范围
        self.max_rad = np.radians(self.max_deg, dtype=np.float32)
        self.max_yz = np.abs(self.max_x * np.sin(self.max_rad), dtype = np.float32)

        # 夹角也需要标准化，在距离最近时夹角应该会有最大值，最大值在(r, 0, 0)或者(r, 0, -self.sqrt3 * r / 6.)
        self.max_iangle = self.cord2rad(Vector3D(self.r, 0, -self.sqrt3 * self.r / 6.), 'max') 
        # 最小值在最远处(max_x, max_yz, max_yz) (max_x, 0, max_yz) 或 (max_x, 0, -max_yz)
        self.min_iangle = self.cord2rad(Vector3D(self.max_x, self.max_yz, self.max_yz), 'min') 
        # print(self.max_iangle, self.min_iangle)

    def _sample_data(self):
        '''
        计算数据集的取值范围，并随机均匀采样一个坐标数据，计算对应的夹角及向量长度。
        '''
        x = np.random.uniform(self.min_x, self.max_x)
        yz = np.abs(x, dtype=np.float32) * np.sin(self.max_rad, dtype=np.float32)
        y = np.random.uniform(-yz, yz)
        z = np.random.uniform(-yz, yz)

        # 定义坐标向量
        cord = Vector3D(x, y, z)
        # pdb.set_trace()
        # cord = Vector3D(18.7504,  9.3123,  5.6839)
        d = cord.magnitude()
        # 夹角计算及标准化
        inc_angle = self.cord2rad(cord)
        # 坐标值标准化
        cord_n = cord.norm(self.max_x, self.r, self.max_yz, -self.max_yz)
        # pdb.set_trace()

        return {'inc_angle': inc_angle, 'cord_n': cord_n.tensor, 'cord': cord.tensor, 'd': d, 
                'max_x': self.max_x, 'min_x': self.min_x, 'max_yz': self.max_yz, 
                'max_iangle': self.max_iangle, 'min_iangle': self.min_iangle}

    def cord2data(self, x, y, z):
        # cord = Vector3D(self.max_x, -self.max_yz, self.max_yz) # 0.0181
        # cord = Vector3D(self.max_x, 0, self.max_yz) # 0.0220
        # cord = Vector3D(self.max_x, 0, -self.max_yz) # 0.220
        # cord = Vector3D(self.r, 0, -self.sqrt3 * self.r / 6.)
        # cord = Vector3D(np.sqrt(6) /3. *self.r, 0, 0)
        cord = Vector3D(x, y, z)
        d = cord.magnitude()
        # 夹角计算
        inc_angle = self.cord2rad(cord)
        # 夹角也需要标准化，在距离最近时夹角应该会有最大值，最大值在(r, 0, 0)或者(r, 0, -self.sqrt3 * r / 3.)
        
        # 坐标值标准化
        cord_n = cord.norm(self.max_x, self.r, self.max_yz, -self.max_yz)

        return {'inc_angle': inc_angle, 'cord_n': cord_n.tensor, 'cord': cord.tensor, 'd': d, 
                'max_x': self.max_x, 'min_x': self.min_x, 'max_yz': self.max_yz, 
                'max_iangle': self.max_iangle, 'min_iangle': self.min_iangle}
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
        self.r = r
        self.max_deg = max_deg
        self.max_x = max_x
        self.vali_max_x = vali_max_x
        self.vali_min_x = vali_min_x
        # self.min_x = min_x
        # self.loc_train = LocData(data_config)
    
    def setup(self, stage=None):
        # self.loc_train = LocData('train', self.r, self.max_deg, self.max_x, self.min_x, self.train_len)
        # self.loc_test  = LocData('test', self.r, self.max_deg, self.max_x, self.min_x, self.test_len)
        # self.loc_val   = LocData('val', self.r, self.max_deg, self.max_x, self.min_x, self.val_len)
        self.loc_train = LocData(self.r, self.max_deg, self.max_x, None, self.train_len)
        self.loc_test  = LocData(self.r, self.max_deg, self.max_x, None, self.test_len)
        self.loc_val   = LocData(self.r, self.max_deg, self.vali_max_x, self.vali_min_x, self.val_len)

    def train_dataloader(self):
        return DataLoader(self.loc_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.loc_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.loc_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.loc_test, batch_size=self.batch_size, num_workers=self.num_workers)
