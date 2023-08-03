# 20230721
## 实验1
### 设置
#### 数据
- data_len 10000
- encoder_list: 
    - {'dim_in': 4,  'dim_ff': 6,  'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 12, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 24, 'dim_out': 32, 'heads': 8, 'dim_head': 16,}
  
  decoder_list: 
    - {'dim_in': 32, 'dim_ff': 24, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 12, 'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 6,  'dim_out': 4,  'heads': 8, 'dim_head': 16,}
  regre_list:
  - 4
  - 1
#### 模型
- encoder_list: 
    - {'dim_in': 4,  'dim_ff': 6,  'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 12, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 24, 'dim_out': 32, 'heads': 8, 'dim_head': 16,}
  
- decoder_list: 
    - {'dim_in': 32, 'dim_ff': 24, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 12, 'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 6,  'dim_out': 4,  'heads': 8, 'dim_head': 16,}
- regre_list:
  - 4
  - 1
### 结果
- Lt_t_m 20
### 总结
1. 增大data_len之后，总的训练损失降低了很多，为26
2. 模型参数$48.1k$，尺寸$0.192MB$

## 实验2
### 设置
#### 数据
- data_len 20000
- encoder_list: 
    - {'dim_in': 4,  'dim_ff': 6,  'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 12, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 24, 'dim_out': 32, 'heads': 8, 'dim_head': 16,}
  
  decoder_list: 
    - {'dim_in': 32, 'dim_ff': 24, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 12, 'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 6,  'dim_out': 4,  'heads': 8, 'dim_head': 16,}
  regre_list:
  - 4
  - 1
#### 模型
- encoder_list: 
    - {'dim_in': 4,  'dim_ff': 6,  'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 12, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 24, 'dim_out': 32, 'heads': 8, 'dim_head': 16,}
  
- decoder_list: 
    - {'dim_in': 32, 'dim_ff': 24, 'dim_out': 16, 'heads': 8, 'dim_head': 16,}
    - {'dim_in': 16, 'dim_ff': 12, 'dim_out': 8,  'heads': 8, 'dim_head': 16,}
    - {'dim_in': 8,  'dim_ff': 6,  'dim_out': 4,  'heads': 8, 'dim_head': 16,}
- regre_list:
  - 4
  - 1
### 结果
- Lt_t_m 63
### 总结
1. lambda_wei_cord权重增加为$2$之后，总的损失只能下降到63
