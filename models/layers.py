import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import einsum
from einops import rearrange, repeat
from functools import wraps
# from pytorch_lightning import LightningModule
# import flash.image 
# from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

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

class LayerNorm(nn.Module):
    '''
    Only norm in the last dim. nn.Layer Norm是作用于样本的每个特征, 一般是最后一个维度
    '''
    def __init__(self, dim_feats, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim_feats))
        self.beta = nn.Parameter(torch.zeros(dim_feats))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
        

class AttBlock(nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        layer_num = len(dim_list)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            layer_setting = dim_list[i]
            self.layers.append(
                AttLayer(
                    dim_in = layer_setting['dim_in'], 
                    dim_ff = layer_setting['dim_ff'], 
                    dim_out = layer_setting['dim_out'], 
                    heads = layer_setting['heads'], 
                    dim_head = layer_setting['dim_head'],
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttLayer(nn.Module):
    '''
    自适应的多头注意力
    '''
    def __init__(self, dim_in = 3, dim_ff = 6, dim_out = 3, heads = 3, dim_head = 12, drop_out=0.):
        super().__init__()
        # 多头注意力机制
        self.self_attn = MultiHeadAttention(
            query_dim=dim_in, 
            heads=heads,
            dim_head=dim_head)
        # 线性变换层
        if dim_in!=1 and dim_in != dim_out:
            self.linear_tran = nn.Linear(dim_in, dim_out)
        # 前馈全连接网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_in , dim_ff),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(dim_ff, dim_out),
        )
        # 残差连接和层归一化
        self.layer_norm1 = nn.LayerNorm(dim_in)
        self.layer_norm2 = nn.LayerNorm(dim_out)
        self.drop_out1 = nn.Dropout(drop_out)
        self.drop_out2 = nn.Dropout(drop_out)
        # self.layer_norm_y = layers.LayerNorm(dim_in)
                
    def forward(self, x):
        # x (b, len_seq, dim_feats)
        y, att_weights_ = self.self_attn(x)
        y = x + self.drop_out1(y)
        y = self.layer_norm1(y)
        # pdb.set_trace()

        # import score
        ff_output = self.feed_forward(y)

        # linear transform
        if hasattr(self, 'linear_tran'):
            y = self.linear_tran(y)

        y = y+ self.drop_out2(ff_output)
        y = self.layer_norm2(y)
        return y
    
class AdpFeedForward(nn.Module):
    def __init__(self, dim_in, dim_ff, dim_out, drop_out=0.):
        super().__init__()
        self.dim_out = dim_out
        # 前馈全连接网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_in , dim_ff),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(dim_ff, dim_out),
        )
        # 线性转换层
        if dim_in!=1 and dim_in != dim_out:
            self.linear_tran = nn.Linear(dim_in, dim_out)

        # 层归一化和Dropout
        self.layer_norm1 = nn.LayerNorm(dim_in)
        self.layer_norm2 = nn.LayerNorm(dim_out)

        self.drop_out1 = nn.Dropout(drop_out)
        self.drop_out2 = nn.Dropout(drop_out)
    
    def forward(self, x):
        # 前馈全连接网络
        ff_output = self.feed_forward(x)

        # linear transform
        if hasattr(self, 'linear_tran'):
            x = self.linear_tran(x)

        x = x+ self.drop_out2(ff_output)
        if self.dim_out != 1:
            x = self.layer_norm2(x)

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out), attn

class RegreHead(nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(dim_list)-1):
            self.layers.append(
                AdpFeedForward(dim_list[i], dim_list[i], dim_list[i+1])
            )

        self.layers.append(nn.Sigmoid())
                    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)                
        return x
    
class DataTransf(nn.Module):
    def __init__(self, dim_in = 1, dim_ff = 6, dim_out = 6, heads = 3, dim_head = 12):
        super().__init__()
        assert dim_in == 1, 'for (b, s, 1) data'
        self.att_layer = AttLayer(
            dim_in = dim_in, 
            dim_ff = dim_ff, 
            dim_out = dim_out, 
            heads = heads, 
            dim_head = dim_head,
        )

    def forward(self, x):
        x = self.att_layer(x)
        return x