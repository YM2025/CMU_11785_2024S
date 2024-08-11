import numpy as np
from scipy.special import erf

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
        # 缓存当前层的激活输出
        self.A = 1 / (1 + np.exp(-Z))
        return self.A;
    
    def backward(self, dLdA):
        dLdZ = dLdA * (self.A - self.A * self.A)
        return dLdZ
    


class Tanh:
    def forward(self, Z):
        # 缓存当前层的激活输出
        self.A = np.tanh(Z)
        return self.A;
    
    def backward(self, dLdA):
        dLdZ = dLdA * (1 - np.square(self.A))
        return dLdZ


class ReLU:
    def forward(self, Z):
        # 缓存当前层的激活输出
        self.A = np.maximum(0, Z)
        return self.A;
    
    def backward(self, dLdA):
        dLdZ = np.where(self.A > 0, dLdA, 0)
        return dLdZ

class GELU:
    def forward(self, Z):
        # 缓存Z以便在反向传播中使用
        self.Z = Z 
        # 缓存当前层的激活输出
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A;
    
    def backward(self, dLdA):
        dAdZ = 0.5 * (1 + erf(self.Z / np.sqrt(2))) + self.Z * np.exp(-np.square(self.Z) / 2) / (np.sqrt(2 * np.pi))
        dLdZ = dLdA * dAdZ
        return dLdZ

class Softmax:
    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A =  e_Z / np.sum(e_Z, axis=1, keepdims=True) 

        return self.A
    
    def backward(self, dLdA):

        """
        计算Softmax层的反向传播。
        
        参数:
        dLdA (np.array): 损失函数关于Softmax输出的梯度，形状为 (N, C)。
        
        返回:
        dLdZ (np.array): 损失函数关于Softmax层输入的梯度，形状为 (N, C)。
        """
        # 计算批量大小和特征数
        N, C = dLdA.shape

        # 初始化最终输出的梯度矩阵dLdZ
        dLdZ = np.zeros_like(dLdA)

        # 对每一个数据点逐一处理
        for i in range(N):
            # 提取第i个样本的Softmax输出
            a = self.A[i, :]
            # 初始化Jacobian矩阵为零矩阵
            J = np.zeros((C, C))

            # 根据上述条件填充Jacobian矩阵
            for m in range(C):
                for n in range(C):
                    if m == n:
                        # 对角线元素：第m个输出相对于第m个输入的导数
                        J[m, n] = a[m] * (1 - a[m])
                    else:
                        # 非对角线元素：第m个输出相对于第n个输入的导数
                        J[m, n] = -a[m] * a[n]

            # 计算损失函数关于第i个输入的导数
            dLdZ[i, :] = np.dot(dLdA[i, :], J)

        return dLdZ
