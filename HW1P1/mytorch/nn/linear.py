import numpy as np

class Linear:
    def __init__(self, in_features, out_features, random_init=False, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        if random_init:
            self.W = np.random.randn(out_features, in_features)
            self.b = np.random.randn(out_features, 1)
        else:
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