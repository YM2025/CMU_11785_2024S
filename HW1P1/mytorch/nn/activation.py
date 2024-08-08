import numpy as np
import scipy
import math



class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A ** 2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    def forward(self, Z):
        # 如果 Z 中的元素小于0，则取0；如果大于等于0，则取元素本身的值
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        # 如果激活值 A 大于 0，梯度为 dLdA；否则，梯度为 0
        dLdZ = np.where(self.A > 0, dLdA, 0)
        return dLdZ


class GELU:
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / math.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        dLdZ = dLdA*(0.5*(1+scipy.special.erf(self.Z/math.sqrt(2)))+(self.Z/math.sqrt(2*math.pi))*np.exp(-self.Z*self.Z/2))
        return dLdZ


class Softmax:
    def forward(self, Z):
        """
        参数:
        Z (np.array): 输入数据，形状为 (N, C)，其中 N 是批量大小，C是类别数。
        
        返回:
        self.A (np.array): 应用Softmax后的输出概率。
        """
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) ## 防止指数运算溢出，通过减去每一行的最大值来进行数值稳定化
        self.A = e_Z / np.sum(e_Z, axis=1, keepdims=True)
        return self.A
    
    def backward(self, dLdA):
        """
        参数:
        dLdA (np.array): 损失函数关于Softmax输出的梯度，形状为 (N, C)。
        
        返回:
        dLdZ (np.array): 损失函数关于Softmax层输入的梯度，形状为 (N, C)。
        """
        N, C = dLdA.shape
        dLdZ = np.zeros_like(dLdA) # 初始化最终输出的梯度矩阵dLdZ

        # 对每一个样本逐一处理
        for i in range(N):
            a = self.A[i, :]      # 提取第i个样本的Softmax输出
            J = np.zeros((C, C))  # 初始化Jacobian矩阵为零矩阵

            # 根据上述条件填充Jacobian矩阵
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = a[m] * (1 - a[m])
                    else:
                        J[m, n] = -a[m] * a[n]

            # 计算损失函数关于第i个输入的导数
            dLdZ[i, :] = np.dot(dLdA[i, :], J)

        return dLdZ