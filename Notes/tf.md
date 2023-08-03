# 位姿回归网络
1. 网络结构
    1. 多层感知机
        - 结构简单，
        - 精度不够，创新不够
    2. Transformer
        - 没有查到回归任务直接用Transformer
        - 我自己尝试理解Transformer并构建网络，发现不能只是在网络后面接一个回归头，输入是序列数据这条无法满足
    3. 多头注意力机制
        - Transformer的核心作用机制，用作构建回归网络，更合理更简单
        - 网络结构
            - DataTransf 数据维度扩充
            - AttentionEncoder
                - AttentionEncoderLayer
                    - MultiHeadAttention
                    - FeedForward
            - AttentionDecoder
                - AttentionDecoderLayer
                    - MultiHeadAttention
                    - FeedForward
            - RegreHead 回归头
            
        - 位置数据扩充网络
            - 由一层AttentionEncoder组成

2. 训练方式
    1. 有监督训练
        三维位置数据生成，添加姿态
    2. 掩码训练
        借鉴于MAE，训练过程中添加噪声，网络更鲁邦

3. 回归损失
    1. 调节各损失权重偏向于yz坐标精度
    2. 位置坐标损失+角度损失+距离损失 $\rightarrow$ yz轴坐标损+角度损失+距离损失







```
x = inputs

for i in range(num_layers):
    x = MultiHeadAttention(num_heads=num_heads, 

    key_dim=hidden_dim)([x, x])

    x = Dropout(dropout_rate)(x)

    x = LayerNormalization(x)

    x = Dense(hidden_dim, activation='relu')(x)

    x = Dropout(dropout_rate)(x)

    x = LayerNormalization(x)

    x = Dense(num_features)(x)

outputs = x
```

