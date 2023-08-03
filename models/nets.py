import pdb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple

from . import layers

class RegreTransf(nn.Module):
    def __init__(self, data_expand, data_dim, encoder_list, decoder_list, regre_list):
        super().__init__()
        self.l = data_dim
        self.data_expand = data_expand
        self.pe = nn.Parameter(torch.eye(self.l), requires_grad=False)
        # heads或者dim_head 增长
        self.att_encoder = layers.AttBlock(encoder_list)
        # 好像并不需要再次进行DataTransf
        # heads或者dim_head 减少
        self.att_decoder = layers.AttBlock(decoder_list)
        # 回归头 RegreHead (b, 3, a) -> (b, 3, 1)
        self.regre_head = layers.RegreHead(regre_list)
    
    def forward(self, x):
        b, l, d = x.shape
        # 坐标编码
        x = torch.cat([x, self.pe.expand((b, self.l, self.l))], dim = -1)
        # 打乱位置编码
        if self.data_expand:
            # 生成一个随机的索引张量
            idx = torch.randperm(l)
            x = x[:, idx, :]
        # 编码
        x = self.att_encoder(x)
        # 解码
        x = self.att_decoder(x)
        # 回归
        x = self.regre_head(x)
  
        return x
    
class MLP(nn.Module):
    def __init__(self, input_size: int = 3, hidden_units: List[int] = [6, 12, 24, 12, 6, 3]):
        super().__init__()
        # Model similar to previous section:

        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units: 
            layer = nn.Linear(input_size, hidden_unit) 
            all_layers.append(layer) 
            all_layers.append(nn.ReLU()) 
            input_size = hidden_unit 
 
        all_layers.append(nn.Linear(hidden_units[-1], 3)) 
        all_layers.append(nn.Sigmoid()) 

        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.model(x)