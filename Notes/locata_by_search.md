# Locate by search


## 方法
- 搜索空间
$30m$, $\pi$, $2\pi$

- 问题建模
1. 设置搜索空间的大小
2. $(d, \phi, \theta)$


## 代码
### gen_data.py

### search_kernel.cu

- cal_pos
计算搜索点的最左上角的顶点位置

- cal_norm
计算该向量的模

- cal_unit
计算该向量的方向

- cal_dot
计算两个向量的点积

