# import pdb

# import torch
# from models.mas import RegreTransf
# encoder_list = []

# encoder_list.append({'dim_in': 4, 
#                  'dim_ff': 6,
#                  'dim_out': 8,
#                  'heads': 8,
#                  'dim_head': 32,
#                  }
#                  )
# encoder_list.append({'dim_in': 8, 
#                  'dim_ff': 12,
#                  'dim_out': 16,
#                  'heads': 8,
#                  'dim_head': 32,
#                  }
#                  )
# encoder_list.append({'dim_in': 16, 
#                  'dim_ff': 24,
#                  'dim_out': 32,
#                  'heads': 8,
#                  'dim_head': 32,
#                  }
#                  )

# decoder_list = []

# decoder_list.append({'dim_in': 32, 
#                  'dim_ff': 24,
#                  'dim_out': 16,
#                  'heads': 8,
#                  'dim_head': 32,
#                  }
#                  )
# decoder_list.append({'dim_in': 16, 
#                  'dim_ff': 12,
#                  'dim_out': 8,
#                  'heads': 8,
#                  'dim_head': 32,
#                  }
#                  )
# decoder_list.append({'dim_in': 8, 
#                  'dim_ff': 6,
#                  'dim_out': 4,
#                  'heads': 8,
#                  'dim_head': 32,
#                  }
#                  )
# regre_list = [4, 3]
# # net = AttBlock(dim_list)
# data_dim=3
# net = RegreTransf(data_dim, encoder_list, decoder_list, regre_list )
# # print(net)

# x = torch.randn(8, 3, 1)

# y = net(x)

# print(y.shape)
# # import pdb
# # from data.loc_data import LocData
# # r = 0.8
# # max_deg = 30
# # max_x = 30
# # min_x = r
# # data_len = 100

# # loc_data = LocData(r, max_deg, max_x, data_len)

# # for i in range(100):
# #     loc_data_sample = loc_data._sample_data()
# #     pdb.set_trace()
# #     print(loc_data_sample['inc_angle'])
 
# # # loc_data_sample = loc_data.cord2data(0,0,0)
# # pdb.set_trace()
# # import torch
# # from models.mas import AttEncoderLayer
# # # from models.mas import VasNet


# # # 创建一个多头注意力层，输入维度为3，输出维度为3，头数为1
# # # mha = MultiHeadAttention(input_dim=3,output_dim=3,num_heads=3)
# # # 创建一个随机输入，批量大小为4，序列长度为5，输入维度为3
# # x = torch.randn(4 ,6 ,3)
# # # 前向传播，得到输出
# # # y = mha(x)
# # # 输出形状为[4 ,5 ,3]，即[批量大小，序列长度，输出维度]
# # # print(y.shape)


# # net = AttEncoderLayer(3, 12, 3, 3, 12)
# # y = net(x)
# # print(y['y'].shape)

# from models.mas import AttEncoder
# import torch

# dim_list = []
# # dim_in_feats = 1
# # dim_ff = 6
# # dim_out_feats = 1
# # heads = 3
# # dim_head = 12
# dim_list.append({'dim_in': 1, 
#                  'dim_ff': 6,
#                  'dim_out': 6,
#                  'heads': 3,
#                  'dim_head': 12,
#                  }
#                  )
# dim_list.append({'dim_in': 6, 
#                  'dim_ff': 6,
#                  'dim_out': 6,
#                  'heads': 6,
#                  'dim_head': 12,
#                  }
#                  )
# dim_list.append({'dim_in': 6, 
#                  'dim_ff': 3,
#                  'dim_out': 1,
#                  'heads': 12,
#                  'dim_head': 12,
#                  }
#                  )
# # dim_list.append({'dim_in': 6, 
# #                  'dim_ff': 6,
# #                  'dim_out': 6,
# #                  'head': 24,
# #                  'dim_head': 12,
# #                  }
# #                  )
# # dim_list.append({'dim_in': 12, 
# #                  'dim_ff': 6,
# #                  'dim_out': 24,
# #                  'head': 3,
# #                  'dim_head': 12,
# #                  }
# #                  )

# net = AttEncoder(dim_list)
# x = torch.randn(4, 3, 1)
# # net
# y = net(x)
# print(y.shape)

# from data.loc_data import LocData
# r = 0.8
# max_deg = 30
# max_x = 30
# min_x = r
# data_len = 100

# loc_data = LocData(r, max_deg, max_x, data_len)

# loc_data_sample = loc_data._sample_data()


from models.nets import MPL1
import torch
x = torch.randn(3, 9)
net = MPL1()
t, y = net(x)
print(t, y)