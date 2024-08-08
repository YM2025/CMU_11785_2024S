import numpy as np


class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        """
        批量归一化层的初始化函数。
        
        :param num_features: 特征的数量
        :param alpha: 用于计算运行均值和方差的指数加权移动平均系数
        """
        self.alpha = alpha  # 指数加权移动平均的系数
        self.eps = 1e-8  # 防止除以零的小数值

        # 初始化可学习参数，缩放因子和位移项
        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))

        # 梯度初始化
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # 运行时均值和方差，训练时更新，推理时使用
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        批量归一化层的前向传播函数。

        :param Z: 输入数据
        :param eval: 是否为推理模式
        :return: 归一化并缩放、位移后的数据
        """
        self.Z = Z  # 输入数据
        self.N = Z.shape[0]  # 数据批次大小

        if not eval:
            # 训练模式
            self.M = np.mean(Z, axis=0)  # 计算均值
            self.V = np.var(Z, axis=0)  # 计算方差
            
            # 更新运行时均值和方差
            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

            # 归一化处理
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
        else:
            # 推理模式
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)

        # 缩放和位移
        self.BZ = self.BW * self.NZ + self.Bb
        return self.BZ

    def backward(self, dLdBZ):
        """
        批量归一化层的后向传播函数。

        :param dLdBZ: 损失函数关于批量归一化后数据的梯度
        :return: 损失函数关于批量归一化前数据的梯度
        """
        # 计算可学习参数的梯度
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        
        # 计算传递给前一层的梯度
        dLdNZ = dLdBZ * self.BW
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * -0.5 * (self.V + self.eps) ** (-3/2), axis=0, keepdims=True)
        dLdM = np.sum(dLdNZ * -1 / np.sqrt(self.V + self.eps), axis=0, keepdims=True) + \
               dLdV * np.mean(-2 * (self.Z - self.M), axis=0, keepdims=True)
        
        # 归一化前数据的梯度
        dLdZ = dLdNZ / np.sqrt(self.V + self.eps) + \
               dLdV * 2 * (self.Z - self.M) / self.N + \
               dLdM / self.N

        return dLdZ