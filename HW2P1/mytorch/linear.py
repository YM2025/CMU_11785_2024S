import numpy as np

class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        初始化权重和偏置为零。
        :param in_features: 输入特征的数量
        :param out_features: 输出特征的数量
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

        self.debug = debug

    def forward(self, A):
        """
        前向传播
        :param A: 输入层数据，形状为 (N, C0)
        :return: 输出层数据，形状为 (N, C1)
        """
        self.A = A
        self.N = A.shape[0]
        self.Ones = np.ones((self.N, 1))
        Z = np.dot(A, self.W.T) + np.dot(self.Ones, self.b.T)
        return Z

    def backward(self, dLdZ):
        """
        反向传播
        :param dLdZ: 关于输出的梯度
        :return: 关于输入的梯度
        """
        dLdA = np.dot(dLdZ, self.W)
        self.dLdW = np.dot(dLdZ.T, self.A)
        self.dLdb = np.dot(dLdZ.T, self.Ones)

        if self.debug:
            self.dLdA = dLdA

        return dLdA



