import numpy as np
import torch
from torch import linalg as LA 
import pdb


class Vector3D: 
    def __init__(self, x, y, z): 
        # 将三个坐标值转换为一个一维的tensor 
        self.tensor = torch.tensor([x, y, z], dtype=torch.float32) 
    def __str__(self):
        return f"({self.tensor[0]}, {self.tensor[1]}, {self.tensor[2]})"

    # 计算两个向量的点积（内积）
    def dot(self, other):
        # 使用torch.dot函数计算两个tensor的点积
        return torch.dot(self.tensor, other.tensor)

    # 计算两个向量的叉积（外积）
    def cross(self, other):
        # 使用torch.cross函数计算两个tensor的叉积，返回一个新的Vector3D对象
        return Vector3D(*torch.cross(self.tensor, other.tensor))

    # 计算向量的模长
    def magnitude(self):
        # 使用torch.norm函数计算tensor的范数，默认为2范数，即欧几里得距离
        return torch.norm(self.tensor)

    def sub(self, other):
        # 使用torch.sub函数计算两个tensor的差，返回一个新的Vector3D对象
        return Vector3D(*torch.sub(self.tensor, other.tensor))

    # 计算两个向量之间的夹角（弧度制）
    def angle_between(self, other):
        # 使用torch.dot函数和torch.norm函数计算两个tensor之间的夹角余弦值，然后使用torch.acos函数计算反余弦值，即夹角
        cos_theta = torch.dot(self.tensor, other.tensor) / (self.magnitude() * other.magnitude())
        return torch.acos(cos_theta)

    def norm(self, max_x, min_x, max_yz, min_yz):
        # 使用torch.sub和torch.div函数对每个坐标值进行归一化，返回一个新的Vector3D对象
        return Vector3D(*(torch.div(torch.sub(self.tensor, torch.tensor([min_x, min_yz, min_yz])), torch.tensor([max_x - min_x, max_yz - min_yz, max_yz - min_yz]))))

    def denorm(self, max_x, min_x, max_yz, min_yz):
        # 使用torch.mul和torch.add函数对每个坐标值进行反归一化，返回一个新的Vector3D对象
        return Vector3D(*(torch.add(torch.mul(self.tensor, torch.tensor([max_x - min_x, max_yz - min_yz, max_yz - min_yz])), torch.tensor([min_x, min_yz, min_yz]))))
    
class Geometry:
    def __init__(self, r):
        super(Geometry, self).__init__()
        self.r = r
        self.pi = np.pi
        self.sqrt3 = np.sqrt(3).astype(np.float32)

        self.A = Vector3D(0.,       0.,  self.sqrt3 * r / 3.)
        self.B = Vector3D(0., - r / 2., -self.sqrt3 * r / 6.)
        self.C = Vector3D(0.,   r / 2., -self.sqrt3 * r / 6.)

        self.max_iangle = None
        self.min_iangle = None

    def cord2rad(self, cord, mode=None):
        '''
            None mode for calculate the inc_angle
            max mode for calculate the max_angle
            min mode for calculate the min_angle
        '''
        DA = cord.sub(self.A)
        DB = cord.sub(self.B)
        DC = cord.sub(self.C)
        # pdb.set_trace()
        # 弧度，如果要算角度，使用np.degrees()
        alpha = DB.angle_between(DC)
        beta =  DA.angle_between(DC)
        gamma = DA.angle_between(DB)
        # # 弧度值标准化
        # alpha_n = (alpha - min_iangle) / (max_iangle - min_iangle)
        # beta_n  = (beta  - min_iangle) / (max_iangle - min_iangle)
        # gamma_n = (gamma - min_iangle) / (max_iangle - min_iangle)
        if mode == 'max':
            self.max_iangle = max(alpha, beta, gamma)
            return self.max_iangle
        elif mode == 'min':
            self.min_iangle = min(alpha, beta, gamma)
            return self.min_iangle
        else:
            if self.max_iangle != None and self.min_iangle != None:
                alpha_n = (alpha - self.min_iangle) / (self.max_iangle - self.min_iangle)
                beta_n  = (beta  - self.min_iangle) / (self.max_iangle - self.min_iangle)
                gamma_n = (gamma - self.min_iangle) / (self.max_iangle - self.min_iangle)
                return {'inc_angle': torch.stack([alpha, beta, gamma], dim=0), 
                        'inc_angle_n': torch.stack([alpha_n, beta_n, gamma_n], dim=0),
                        'max_iangle': self.max_iangle,
                        'min_iangle': self.min_iangle,}
            else:
                return {'inc_angle': torch.stack([alpha, beta, gamma], dim=0)}
                       

# 定义旋转类
class Rotation:
    def __init__(self, axis, angle_degrees):
        self.axis = axis
        self.angle_rad = np.radians(angle_degrees)  # 将角度转换为弧度

    # 将一个向量绕着指定轴旋转一定角度
    def rotate(self, vector):
        v = np.array([vector.x, vector.y, vector.z])
        k = np.array([self.axis.x, self.axis.y, self.axis.z])
        cos_theta = np.cos(self.angle_rad)
        sin_theta = np.sin(self.angle_rad)

        # 计算旋转矩阵
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        I = np.identity(3)
        R = I * cos_theta + (1 - cos_theta) * np.outer(k, k) + sin_theta * K

        # 计算旋转后的向量
        new_v = np.dot(R, v)
        return Vector3D(*new_v)


# # 示例
# v = Vector3D(1, 0, 0)
# axis = Vector3D(0, 0, 1)
# angle_degrees = 90

# rotation = Rotation(axis, angle_degrees)
# rotated_vector = rotation.rotate(v)
# print(rotated_vector)  # 输出: (6.123233995736766e-17, 1.0, 0.0)

# v1 = Vector3D(1, 2, 3)
# v2 = Vector3D(4, 5, 6)
# angle = v1.angle_between(v2)
# print(angle)  # 输出: 12.933154491899135

