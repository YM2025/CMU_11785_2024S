import numpy as np

class BatchNorm2d:
    def __init__(self, num_features, alpha=0.9):
        # 初始化批量归一化层
        # num_features: 特征数，即通道数
        # alpha: 用于运行时均值和方差的移动平均系数
        self.alpha = alpha  # 移动平均的衰减因子
        self.eps = 1e-8  # 用于防止除以零的小量

        # 归一化后的值、缩放后的值、偏移后的值
        self.Z = None  # 原始输入
        self.NZ = None  # 归一化后的值
        self.BZ = None  # 缩放和偏移后的值

        # 权重和偏置
        self.BW = np.ones((1, num_features, 1, 1))  # 缩放权重
        self.Bb = np.zeros((1, num_features, 1, 1))  # 偏置

        # 梯度
        self.dLdBW = np.zeros((1, num_features, 1, 1))  # 权重梯度
        self.dLdBb = np.zeros((1, num_features, 1, 1))  # 偏置梯度

        # 均值和方差
        self.M = np.zeros((1, num_features, 1, 1))  # 当前批次的均值
        self.V = np.ones((1, num_features, 1, 1))  # 当前批次的方差

        # 运行时均值和方差
        self.running_M = np.zeros((1, num_features, 1, 1))  # 运行时均值
        self.running_V = np.ones((1, num_features, 1, 1))  # 运行时方差

    def __call__(self, *args, **kwargs):
        # 类实例可以像函数一样被调用，内部调用forward方法
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        前向传播：
        参数:
          Z (np.array): 输入数据
          eval (boolean): 是否处于评估模式
        """
        if eval:
            # 评估模式下，使用运行时均值和方差进行归一化
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = self.BW * NZ + self.Bb
            return BZ

        # 保存原始输入
        self.Z = Z
        # 计算当前批次中所有元素的总数
        self.N = Z.shape[0] * Z.shape[2] * Z.shape[3]

        # 计算当前批次的均值和方差
        self.M = np.mean(Z, axis=(0, 2, 3), keepdims=True)
        self.V = np.var(Z, axis=(0, 2, 3), keepdims=True)
        # 根据均值和方差进行归一化
        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
        # 应用权重和偏置进行缩放和偏移
        self.BZ = self.BW * self.NZ + self.Bb

        # 更新运行时均值和方差
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        return self.BZ

    def backward(self, dLdBZ):
        """
        反向传播：
        参数:
          dLdBZ (np.array): 上游传递的梯度
        """
        # 计算关于偏置的梯度
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)
        # 计算关于权重的梯度
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3), keepdims=True)

        # 根据权重计算输入归一化的梯度
        dLdNZ = dLdBZ * self.BW
        # 计算关于方差的梯度
        dLdV = -0.5 * np.sum(dLdNZ * (self.Z - self.M) * ((self.V + self.eps) ** (-1.5)), axis=(0, 2, 3), keepdims=True)

        # 计算均值对归一化的影响
        dNZdM = -(self.V + self.eps) ** (-0.5) - 0.5 * (self.Z - self.M) * (self.V + self.eps) ** (-1.5) * (-2 / self.N * np.sum(self.Z - self.M, axis=(0, 2, 3), keepdims=True))

        # 计算关于均值的梯度
        dLdM = np.sum(dLdNZ * dNZdM, axis=(0, 2, 3), keepdims=True)
        # 综合计算输入数据的梯度
        dLdZ = dLdNZ * (self.V + self.eps) ** (-0.5) + dLdV * (2 / self.N * (self.Z - self.M)) + 1 / self.N * dLdM

        return dLdZ
