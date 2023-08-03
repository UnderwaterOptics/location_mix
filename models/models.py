from typing import Any, Optional
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from pytorch_lightning import LightningModule
from torch import nn 
from typing import List, Tuple

from loss.geo_loss import GeoLoss, Vector3D
from models.nets import RegreTransf, MLP

import pdb


class RegreTransfL(LightningModule):
    def __init__(
            self, 
            r: float = 0.8,
            data_expand: bool = True,
            data_dim: int = 3,
            encoder_list: list = [],
            decoder_list: list = [],
            regre_list: list = [4, 3],
            lr: float = 1e-6,
            lambda_axis: list = [0., 1., 1.],
            lambda_wei_cord: float = 1.,
            lambda_cord: float = 1., 
            lambda_rad: float = 1., 
            lambda_d: float = 1.
        ):
        super().__init__()
        self.save_hyperparameters()
        # Net initialize
        self.rt = RegreTransf(data_expand, data_dim, encoder_list, decoder_list, regre_list) 

        self.geo_loss = GeoLoss(r)

    def forward(self, x):
        return self.rt(x)
    
    def training_step(self, batch, batch_idx):
        # prepare input inc_angle data
        inc_angle_n = batch['inc_angle']['inc_angle_n'].unsqueeze(dim=2)

        # forward process
        cord_np = self(inc_angle_n)
        cord_np = cord_np.squeeze(dim=2)
        # output data Denormalization
        cord_p = torch.stack([
            cord_np[:, 0] * (batch['max_x'] - batch['min_x']) + batch['min_x'],
            cord_np[:, 1] * (2 * batch['max_yz']) - batch['max_yz'],
            cord_np[:, 2] * (2 * batch['max_yz']) - batch['max_yz'],
        ], dim = 1)

        # geo loss
        wei_cord_loss, cord_loss, rad_loss, d_loss = self.geo_loss(cord_p, batch, self.hparams.lambda_axis, True)
        # if torch.isnan(cord_np).any():
        #     pdb.set_trace()

        total_loss = self.hparams.lambda_wei_cord * wei_cord_loss + self.hparams.lambda_cord * cord_loss + self.hparams.lambda_rad * rad_loss + self.hparams.lambda_d * d_loss

        # Record these loss in tensorboard.
        # self.log('train_rad_loss', rad_loss*self.hparams.lambda_rad)
        # self.log('train_d_loss', d_loss)
        # self.log("train_cord_loss", cord_loss)
        # self.log('train_wei_cord_loss', wei_cord_loss, prog_bar=True)
        # self.log('train_total_loss', total_loss)

        return total_loss 

    def validation_step(self, batch, batch_idx):
        # prepare input inc_angle data
        inc_angle_n = batch['inc_angle']['inc_angle_n'].unsqueeze(dim=2)

        # forward process
        cord_np = self(inc_angle_n)
        cord_np = cord_np.squeeze(dim=2)
        # output data Denormalization
        cord_p = torch.stack([
            cord_np[:, 0] * (batch['max_x'] - batch['min_x']) + batch['min_x'],
            cord_np[:, 1] * (2 * batch['max_yz']) - batch['max_yz'],
            cord_np[:, 2] * (2 * batch['max_yz']) - batch['max_yz'],
        ], dim = 1)


        # geo loss
        wei_cord_loss, cord_loss, rad_loss, d_loss = self.geo_loss(cord_p, batch, self.hparams.lambda_axis, False)

        total_loss = self.hparams.lambda_wei_cord * wei_cord_loss + self.hparams.lambda_cord * cord_loss + self.hparams.lambda_rad * rad_loss + self.hparams.lambda_d * d_loss

        # Record these loss in tensorboard.
        self.log('vali_rad_loss', rad_loss*self.hparams.lambda_rad)
        self.log('vali_d_loss', d_loss)
        self.log("vali_cord_loss", cord_loss)
        # self.log("vali_wei_cord_loss", wei_cord_loss)
        self.log('vali_total_loss', total_loss)
        # print(wei_cord_loss)

        return wei_cord_loss 
    

    def test_step(self, batch, batch_idx):
        # prepare input inc_angle data
        inc_angle_n = batch['inc_angle']['inc_angle_n'].unsqueeze(dim=2)

        # forward process
        cord_np = self(inc_angle_n)
        cord_np = cord_np.squeeze(dim=2)
        # output data Denormalization
        cord_p = torch.stack([
            cord_np[:, 0] * (batch['max_x'] - batch['min_x']) + batch['min_x'],
            cord_np[:, 1] * (2 * batch['max_yz']) - batch['max_yz'],
            cord_np[:, 2] * (2 * batch['max_yz']) - batch['max_yz'],
        ], dim = 1)

        # geo loss
        wei_cord_loss, cord_loss, rad_loss, d_loss = self.geo_loss(cord_p, batch, self.hparams.lambda_axis, False)

        total_loss = self.hparams.lambda_wei_cord * wei_cord_loss + self.hparams.lambda_cord * cord_loss + self.hparams.lambda_rad * rad_loss + self.hparams.lambda_d * d_loss

        # Record these loss in tensorboard.
        self.log('test_rad_loss', rad_loss*self.hparams.lambda_rad)
        self.log('test_d_loss', d_loss)
        self.log("test_cord_loss", cord_loss)
        self.log("test_wei_cord_loss", wei_cord_loss)

        return wei_cord_loss 
    
    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        loss_sum = 0
        for i, item in enumerate(outputs):
            loss_sum += item['loss'] 
        loss_mean = loss_sum/len(outputs)

        self.log('Lt_t_m', loss_mean, prog_bar=True)
        # return loss_mean

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        loss_sum = 0
        for i, item in enumerate(outputs):
            loss_sum += item['loss'] 
        loss_mean = loss_sum/len(outputs)

        self.log('Le_wc_m', loss_mean)
        # return loss_mean
    
    def validate_epoch_end(self, outputs: EPOCH_OUTPUT):
        loss_sum = 0
        for i, item in enumerate(outputs):
            loss_sum += item['loss'] 
        loss_mean = loss_sum/len(outputs)

        self.log('Lv_wc_m', loss_mean)
        # return loss_mean
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        return optimizer



class RegreMPL(LightningModule):
    def __init__(self, 
            r: float = 0.8, 
            input_size: int = 3, hidden_units: List[int] = [6, 12, 24, 12, 6, 3],
            lr: float = 1e-5, 
            lambda_axis: list = [0., 1., 1.], 
            lambda_wei_cord: float = 1.,
            lambda_cord: float = 1., 
            lambda_rad: float = 1., 
            lambda_d: float = 1.):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = MLP(input_size, hidden_units)

        # self.loss = nn.MSELoss()
        self.geo_loss = GeoLoss(r)

    def forward(self, x):
        # pdb.set_trace()
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        # prepare input inc_angle data
        inc_angle_n = batch['inc_angle']['inc_angle_n'].unsqueeze(dim=2)

        # forward process
        cord_np = self(inc_angle_n)
        # cord_np = cord_np.squeeze(dim=2)
        # output data Denormalization
        cord_p = torch.stack([
            cord_np[:, 0] * (batch['max_x'] - batch['min_x']) + batch['min_x'],
            cord_np[:, 1] * (2 * batch['max_yz']) - batch['max_yz'],
            cord_np[:, 2] * (2 * batch['max_yz']) - batch['max_yz'],
        ], dim = 1)

        # geo loss
        wei_cord_loss, cord_loss, rad_loss, d_loss = self.geo_loss(cord_p, batch, self.hparams.lambda_axis, True)
        # if torch.isnan(cord_np).any():

        total_loss = self.hparams.lambda_wei_cord * wei_cord_loss + self.hparams.lambda_cord * cord_loss + self.hparams.lambda_rad * rad_loss + self.hparams.lambda_d * d_loss

        # Record these loss in tensorboard.
        # self.log('train_rad_loss', rad_loss*self.hparams.lambda_rad)
        # self.log('train_d_loss', d_loss)
        # self.log("train_cord_loss", cord_loss)
        # self.log('train_total_loss', total_loss)
        # self.log('train_wei_cord_loss', wei_cord_loss, prog_bar=True)

        return total_loss 

    
    def validation_step(self, batch, batch_idx):
        # prepare input inc_angle data
        inc_angle_n = batch['inc_angle']['inc_angle_n'].unsqueeze(dim=2)

        # forward process
        cord_np = self(inc_angle_n)
        # cord_np = cord_np.squeeze(dim=2)
        # output data Denormalization
        cord_p = torch.stack([
            cord_np[:, 0] * (batch['max_x'] - batch['min_x']) + batch['min_x'],
            cord_np[:, 1] * (2 * batch['max_yz']) - batch['max_yz'],
            cord_np[:, 2] * (2 * batch['max_yz']) - batch['max_yz'],
        ], dim = 1)


        # geo loss
        wei_cord_loss, cord_loss, rad_loss, d_loss = self.geo_loss(cord_p, batch, self.hparams.lambda_axis, False)

        total_loss = self.hparams.lambda_wei_cord * wei_cord_loss + self.hparams.lambda_cord * cord_loss + self.hparams.lambda_rad * rad_loss + self.hparams.lambda_d * d_loss

        # Record these loss in tensorboard.
        self.log('vali_rad_loss', rad_loss*self.hparams.lambda_rad)
        self.log('vali_d_loss', d_loss)
        self.log("vali_cord_loss", cord_loss)
        # self.log("vali_wei_cord_loss", wei_cord_loss, prog_bar=True)
        self.log('vali_total_loss', total_loss)
        # print(wei_cord_loss)

        return wei_cord_loss 


    def test_step(self, batch, batch_idx):
        # prepare input inc_angle data
        inc_angle_n = batch['inc_angle']['inc_angle_n'].unsqueeze(dim=2)

        # forward process
        cord_np = self(inc_angle_n)
        # cord_np = cord_np.squeeze(dim=2)
        # output data Denormalization
        cord_p = torch.stack([
            cord_np[:, 0] * (batch['max_x'] - batch['min_x']) + batch['min_x'],
            cord_np[:, 1] * (2 * batch['max_yz']) - batch['max_yz'],
            cord_np[:, 2] * (2 * batch['max_yz']) - batch['max_yz'],
        ], dim = 1)

        # geo loss
        wei_cord_loss, cord_loss, rad_loss, d_loss = self.geo_loss(cord_p, batch, self.hparams.lambda_axis, False)

        total_loss = self.hparams.lambda_wei_cord * wei_cord_loss + self.hparams.lambda_cord * cord_loss + self.hparams.lambda_rad * rad_loss + self.hparams.lambda_d * d_loss

        # Record these loss in tensorboard.
        self.log('test_rad_loss', rad_loss*self.hparams.lambda_rad)
        self.log('test_d_loss', d_loss)
        self.log("test_cord_loss", cord_loss)
        self.log("test_wei_cord_loss", wei_cord_loss, prog_bar=True)
        self.log('test_cord_loss', total_loss)

        return wei_cord_loss 
    
    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        loss_sum = 0
        for i, item in enumerate(outputs):
            loss_sum += item['loss'] 
        loss_mean = loss_sum/len(outputs)

        self.log('Lt_t_m', loss_mean, prog_bar=True)
        # return loss_mean

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        loss_sum = 0
        for i, item in enumerate(outputs):
            loss_sum += item['loss'] 
        loss_mean = loss_sum/len(outputs)

        self.log('Le_wc_m', loss_mean)
        # return loss_mean
    
    def validate_epoch_end(self, outputs: EPOCH_OUTPUT):
        loss_sum = 0
        for i, item in enumerate(outputs):
            loss_sum += item['loss'] 
        loss_mean = loss_sum/len(outputs)

        self.log('Lv_wc_m', loss_mean)
        # return loss_mean

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        return optimizer

