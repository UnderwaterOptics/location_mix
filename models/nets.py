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
    def __init__(self, input_size: int = 9, hidden_units: List[int] = [6, 12, 24, 12, 6, 3]):
        super().__init__()
        # Model similar to previous section:

        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units: 
            layer = nn.Linear(input_size, hidden_unit) 
            all_layers.append(layer) 
            all_layers.append(nn.ReLU()) 
            input_size = hidden_unit 
 
        all_layers.append(nn.Linear(hidden_units[-1], 4)) 
        all_layers.append(nn.Sigmoid()) 

        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.model(x)
    

class MPL1(nn.Module):
    def __init__(self, 
                 input_size: int = 9, 
                 hidden_sizes: List[int] = [64, 256, 512,],
                 output_sizes0: List[int] = [256, 128], 
                 output_sizes1: List[int] = [256, 128], 
                 output_size0: int = 3,
                 output_size1: int = 1,
                 ):
        super(MPL1, self).__init__()

        hidden_layers = []
        for hidden_unit in hidden_sizes:
            layer = nn.Linear(input_size, hidden_unit)
            hidden_layers.append(layer)
            hidden_layers.append(nn.ReLU())
            input_size = hidden_unit
        self.hidden_layers = nn.Sequential(*hidden_layers)

        output_layers0 = []
        input_size = hidden_sizes[-1]
        for output_unit in output_sizes0:
            layer = nn.Linear(input_size, output_unit)
            output_layers0.append(layer)
            output_layers0.append(nn.ReLU())
            input_size = output_unit
        output_layers0.append(nn.Linear(output_sizes0[-1], output_size0))
        output_layers0.append(nn.Tanh())
        self.output_layers0 = nn.Sequential(*output_layers0)
        
        output_layers1 = []
        input_size = hidden_sizes[-1]
        for output_unit in output_sizes1:
            layer = nn.Linear(input_size, output_unit)
            output_layers1.append(layer)
            output_layers1.append(nn.ReLU())
            input_size = output_unit
        output_layers1.append(nn.Linear(output_sizes1[-1], output_size1))
        output_layers1.append(nn.Sigmoid())
        self.output_layers1 = nn.Sequential(*output_layers1)
        
    def forward(self, x):
        x = self.hidden_layers(x)

        t = self.output_layers0(x)
        d = self.output_layers1(x)
        
        return t, d