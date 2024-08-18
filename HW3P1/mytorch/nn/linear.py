import numpy as np

class Linear:
    
    def __init__(self, in_features, out_features, debug=False):
        """
        初始化线性层

        参数:
        in_features  (int): 输入特征的数量
        out_features (int): 输出特征的数量
        debug        (bool): 是否开启调试模式，默认关闭
        """

        # 初始化权重矩阵 W 和偏置向量 b 为零矩阵
        self.W    = np.zeros((out_features, in_features), dtype="f")
        self.b    = np.zeros((out_features, 1), dtype="f")
        # 初始化梯度 dLdW 和 dLdb 为零矩阵，用于反向传播时存储梯度
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        
        self.debug = debug  # 是否开启调试模式

    def __call__(self, A):
        """
        使得类的实例可以像函数一样被调用，调用时执行 forward 方法
        """
        return self.forward(A)
        
    def forward(self, A):
        """
        前向传播：计算线性层的输出 Z

        参数:
        A (np.array): 输入数据矩阵，维度为 (batch_size, in_features)

        返回:
        Z (np.array): 线性层的输出，维度为 (batch_size, out_features)
        """
    
        self.A    = A  # 保存输入矩阵 A
        self.N    = A.shape[0]  # 获取 batch_size
        self.Ones = np.ones((self.N, 1), dtype="f")  # 构造一个全为1的列向量，维度为 (batch_size, 1)
        Z         = self.A @ self.W.T + self.Ones @ self.b.T  # 计算输出 Z
        
        return Z
        
    def backward(self, dLdZ):
        """
        反向传播：计算损失函数对输入、权重和偏置的梯度

        参数:
        dLdZ (np.array): 损失函数对输出 Z 的梯度，维度为 (batch_size, out_features)

        返回:
        dLdA (np.array): 损失函数对输入 A 的梯度，维度为 (batch_size, in_features)
        """
    
        dZdA      = self.W.T  # 计算 Z 对 A 的梯度，即 W 的转置
        dZdW      = self.A    # 计算 Z 对 W 的梯度，即 A
        dZdi      = None      # 计算 Z 对输入的梯度，未使用，留空
        dZdb      = self.Ones # 计算 Z 对 b 的梯度，即全1列向量
        dLdA      = dLdZ @ dZdA.T  # 计算损失函数对输入 A 的梯度
        dLdW      = dLdZ.T @ dZdW  # 计算损失函数对权重 W 的梯度
        dLdi      = None           # 计算损失函数对输入的梯度，未使用，留空
        dLdb      = dLdZ.T @ dZdb  # 计算损失函数对偏置 b 的梯度
        self.dLdW = dLdW           # 存储计算得到的梯度 dLdW
        self.dLdb = dLdb           # 存储计算得到的梯度 dLdb

        if self.debug:
            # 如果 debug 模式开启，保存中间计算值以供调试
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi
        
        return dLdA  # 返回损失函数对输入 A 的梯度
