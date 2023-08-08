# Mix Location
Mix location 由位置回归网络和位置搜索算法两部分组成。
## 位置回归网络
### 数据生成
1. 按照位置搜索算法的描述进行问题建模
2. 数据流
    1. 在搜索空间中采样产生两个坐标之间的平移数据
    2. 表示三个导引灯在AUV坐标系中的方向向量
    3. 网络输入三个灯的方向向量，输出平移坐标

## 位置搜索算法
### 问题建模
1. 建立坐标系
    1. 在光信标处建立右手坐标系，$x$轴垂直纸面向里，$y$轴水平向左，$z$轴竖直向上，此时，三个光信标的坐标分别为：$A (0, 0, -\frac{\sqrt{3}r}{3})$, $B (0, -\frac{r}{2}, -\frac{\sqrt{3}r}{6})$, $C (0, \frac{r}{2}, -\frac{\sqrt{3}r}{6})$；
    2. 在AUV处建立同样的右手坐标系，$x$轴指向AUV机头方向，$y$轴垂直机身向左，$z$轴竖直机身向上。当AUV机头正朝向光信标时，AUV坐标系与光信标坐标系的坐标轴方向是一致的。
2. 坐标变化
    1. 根据刚体变换公式，设置由光信标坐标系到AUV坐标系的旋转矩阵为$R$，平移向量为$T$，有：
        $$
        P_i^{\prime}=\left(\begin{array}{c}
        x_i{ }^{\prime} \\
        y_i{ }^{\prime} \\
        z_i{ }^{\prime}
        \end{array}\right)=R P_i+T=R\left(\begin{array}{c}
        x_i \\
        y_i \\
        z_i
        \end{array}\right)+T.
        $$
        参考空气动力学，仅考虑偏航角$\theta_y$作姿态矩阵：
        $$
        R=R_y\left(\theta_y\right)=\left[\begin{array}{ccc}
        \cos \theta_y & 0 & \sin \theta_y \\
        0 & 1 & 0 \\
        -\sin \theta_y & 0 & \cos \theta_y
        \end{array}\right]
        $$
        平移向量：
        $$
        T=\left[\begin{array}{l}
        t_x \\
        t_y \\
        t_z
        \end{array}\right]
        $$

    2. 正向转换方程为：
        $$
        \begin{aligned}
        \overrightarrow{O P_i} & =\left[\begin{array}{ccc}
        \cos \theta_y & 0 & \sin \theta_y \\
        0 & 1 & 0 \\
        -\sin \theta_y & 0 & \cos \theta_y
        \end{array}\right]\left(\begin{array}{l}
        x_i \\
        y_i \\
        z_i
        \end{array}\right)+\left(\begin{array}{l}
        t_x \\
        t_y \\
        t_z
        \end{array}\right) \\
        \overrightarrow{e_i} & =\frac{1}{\left\|O P_i{ }^{\prime}\right\|} \overrightarrow{O P_i{ }^{\prime}}
        \end{aligned}
        $$
3. 问题建模为已知三个方向向量$\bar{e}_1$、$\bar{e}_2$和$\bar{e}_3$，逆向求取参量$\theta_y$、$t_x$、 $t_y$和$t_z$。
4. 该问题暂不考虑航向角，问题简化为已知三个方向向量$\bar{e}_1$、$\bar{e}_2$和$\bar{e}_3$，逆向求取参量$t_x$、 $t_y$和$t_z$。
### 搜索空间
1. 坐标向量输入

    给定位置坐标的输入向量$E=\left(\begin{array}{lll}\overrightarrow{e_1} & \overrightarrow{e_2} & \overrightarrow{e_3}\end{array}\right)$ 

2. 定义搜索空间

    搜索空间$Y=\left\{\left(\theta_y, t_x, t_y, t_z\right) \mid \theta_y \in \Theta ;\left(t_x, t_y, t_z\right) \in \Psi\right\}$

3. 设定余弦损失函数

    $\operatorname{Loss}\left(E, E^{\prime}\right)=\sum_{i=1}^3 \frac{\overrightarrow{e_i} \cdot \overrightarrow{e_i}{ }^{\prime}}{\left\|\overrightarrow{e_i}\right\|\left\|\vec{e}_i{ }^{\prime}\right\|}=\sum_{i=1}^3 \overrightarrow{e_i} \cdot \overrightarrow{e_i}{ }^{\prime}$

### 搜索条件
1. 设定误差限 $\varepsilon$，如果满足 $\left|\operatorname{Loss}\left(E, E^{\prime}\right)\right|<\varepsilon$，即认为解$y=y^{\prime}$。
**实际的误差其实不是两组向量的余弦损失形式，而是两组坐标的坐标误差**

### 代码实现

#### 数据生成
1. 采样得到平移向量
2. 转换三个导引灯的坐标从光信标坐标系到AUV坐标系
3. 计算三个光信标的方向向量

#### 搜索过程
1. 读取三个光信标的方向向量及平移向量
2. 搜索(x0:(x0+2), y0:(y0+2), z0:(z0+2))的区域，分辨率为dr
3. 每一次搜索都是在一个一米大小的搜索空间中

问题：
1. 搜索空间大小？
2. 搜索次数？

    ```CPP
    # 搜索(x0:(x0+2), y0:(y0+2), z0:(z0+2))的区域，分辨率为dr，输入量e1,e2,e3,局部最优解ans
    void search(Point e1, Point e2, Point e3, int x0, int y0, int z0, Ans& ans, Bound bound,    Resolution resolution, Ans* ans_cuda)
    ```


