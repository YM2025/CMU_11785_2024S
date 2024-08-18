import numpy as np
from nn.activation import *

class RNNCell(object):
    """RNN 单元类"""

    def __init__(self, input_size, hidden_size):
        """
        初始化 RNNCell 类

        参数:
        ----
        input_size: 输入的维度大小
        hidden_size: 隐藏层的维度大小
        """

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 激活函数为 Tanh
        self.activation = Tanh()

        # 隐藏层维度和输入维度
        h = self.hidden_size
        d = self.input_size

        # 初始化权重和偏置
        self.W_ih = np.random.randn(h, d)  # 输入到隐藏层的权重矩阵
        self.W_hh = np.random.randn(h, h)  # 隐藏层到隐藏层的权重矩阵
        self.b_ih = np.random.randn(h)     # 输入到隐藏层的偏置
        self.b_hh = np.random.randn(h)     # 隐藏层到隐藏层的偏置

        # 初始化梯度
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        """
        初始化权重和偏置

        参数:
        ----
        W_ih: 输入到隐藏层的权重矩阵
        W_hh: 隐藏层到隐藏层的权重矩阵
        b_ih: 输入到隐藏层的偏置
        b_hh: 隐藏层到隐藏层的偏置
        """
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        """
        将所有的梯度初始化为零
        """
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        """
        实现类的可调用性，直接调用 forward 方法

        参数:
        ----
        x: 当前时间步的输入
        h_prev_t: 上一个时间步的隐藏状态

        返回:
        ----
        当前时间步的隐藏状态
        """
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN 单元的前向传播（单个时间步）

        参数:
        ----
        x: (batch_size, input_size) 当前时间步的输入
        h_prev_t: (batch_size, hidden_size) 上一个时间步的隐藏状态

        返回:
        ----
        h_t: (batch_size, hidden_size) 当前时间步的隐藏状态
        """

        # 计算隐藏状态，公式 h_t = tanh(W_ih * x + b_ih + W_hh * h_prev_t + b_hh)
        z = (np.matmul(self.W_ih, np.transpose(x)) + np.expand_dims(self.b_ih, axis=1) + 
             np.matmul(self.W_hh, np.transpose(h_prev_t)) + np.expand_dims(self.b_hh, axis=1))
        h_t = self.activation.forward(np.transpose(z))  # 对 z 应用tanh激活函数
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN 单元的反向传播（单个时间步）

        参数:
        ----
        delta: (batch_size, hidden_size) 当前隐藏层的梯度
        h_t: (batch_size, hidden_size) 当前时间步和当前层的隐藏状态
        h_prev_l: (batch_size, input_size) 当前时间步和前一层的隐藏状态
        h_prev_t: (batch_size, hidden_size) 上一个时间步和当前层的隐藏状态

        返回:
        ----
        dx: (batch_size, input_size) 相对于当前时间步和前一层的输入梯度
        dh_prev_t: (batch_size, hidden_size) 相对于上一个时间步和当前层的输入梯度
        """
        batch_size = delta.shape[0]
        # 0) 反向传播通过 tanh 激活函数。
        dz = self.activation.backward(delta, state=h_t)

        # 1) 计算权重和偏置的梯度
        # 使用 += 的是因为RNN结构在前向传播时会自循环很多次，假设自循环5次，
        # 那么反向传播时就需要调用5次backward，梯度值累加5次后才去更新一次参数。
        self.dW_ih += ((dz.T @ h_prev_l)/ batch_size)
        self.dW_hh += ((dz.T @ h_prev_t) / batch_size)
        self.db_ih += (np.sum(dz, axis=0) / batch_size)
        self.db_hh += (np.sum(dz, axis=0) / batch_size)

        # 2) 计算 dx 和 dh_prev_t
        dx = dz @ self.W_ih
        dh_prev_t = dz @ self.W_hh

        # 3) 返回 dx 和 dh_prev_t
        return dx, dh_prev_t
    
    
    
