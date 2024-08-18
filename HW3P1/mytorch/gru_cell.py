import numpy as np
from nn.activation import *

class GRUCell(object):
    """GRU 单元类。"""

    def __init__(self, input_size, hidden_size):
        """
        初始化 GRU 单元的参数。

        参数:
        ---------
        input_size: int
            输入的维度大小。
        hidden_size: int
            隐藏层的维度大小。
        """
        self.d = input_size  # 输入的维度大小
        self.h = hidden_size  # 隐藏层的维度大小
        h = self.h
        d = self.d
        self.x_t = 0  # 初始化输入

        # GRU 单元的权重矩阵，使用随机数初始化
        self.Wrx = np.random.randn(h, d)  # 重置门的输入到隐藏层的权重
        self.Wzx = np.random.randn(h, d)  # 更新门的输入到隐藏层的权重
        self.Wnx = np.random.randn(h, d)  # 候选隐藏状态的输入到隐藏层的权重

        self.Wrh = np.random.randn(h, h)  # 重置门的隐藏层到隐藏层的权重
        self.Wzh = np.random.randn(h, h)  # 更新门的隐藏层到隐藏层的权重
        self.Wnh = np.random.randn(h, h)  # 候选隐藏状态的隐藏层到隐藏层的权重

        # GRU 单元的偏置，使用随机数初始化
        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        # 初始化各个权重和偏置的梯度为0
        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        # 激活函数
        self.r_act = Sigmoid()  # 重置门的激活函数
        self.z_act = Sigmoid()  # 更新门的激活函数
        self.h_act = Tanh()     # 候选隐藏状态的激活函数

        # 定义其他变量，用于存储前向传播的结果以便反向传播时使用

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        """
        初始化 GRU 单元的权重和偏置。

        参数:
        ---------
        Wrx, Wzx, Wnx, Wrh, Wzh, Wnh: np.array
            对应的权重矩阵。
        brx, bzx, bnx, brh, bzh, bnh: np.array
            对应的偏置向量。
        """
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        # 使对象实例可以像函数一样调用，实际调用的是 forward 方法
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        GRU 单元的前向传播。

        参数:
        ---------
        x: np.array, 形状为 (input_dim)
            当前时间步的输入。
        h_prev_t: np.array, 形状为 (hidden_dim)
            前一个时间步的隐藏状态。

        返回:
        ---------
        h_t: np.array, 形状为 (hidden_dim)
            当前时间步的隐藏状态。
        """
        self.x = x  # 存储输入
        self.hidden = h_prev_t  # 存储前一时间步的隐藏状态
        
        # 计算重置门
        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        # 计算更新门
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        # 计算候选隐藏状态
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))
        # 计算当前时间步的隐藏状态
        h_t = (1 - self.z) * self.n + self.z * h_prev_t
        
        # 确保输入和隐藏状态的维度正确
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t 是 GRU 单元的最终输出

        self.h_t = h_t  # 存储当前时间步的隐藏状态
        return h_t  # 返回当前时间步的隐藏状态

    def backward(self, delta):
        """
        GRU 单元的反向传播。

        计算各参数的梯度，并返回相对于输入 xt 和 ht 的梯度。

        参数:
        ---------
        delta: np.array, 形状为 (hidden_dim)
            损失函数相对于当前时间步的输出以及下一个时间步的梯度之和。

        返回:
        ---------
        dx: np.array, 形状为 (input_dim)
            损失函数相对于输入 x 的梯度。

        dh_prev_t: np.array, 形状为 (hidden_dim)
            损失函数相对于前一时间步隐藏状态 h 的梯度。
        """
        # 计算更新门 z 的梯度
        dz = delta * (-self.n + self.hidden)
        # 计算候选隐藏状态 n 的梯度
        dn = delta * (1 - self.z)

        # 计算 tanh 激活函数的反向传播
        dtanh = self.h_act.backward(dn, state=self.n)
        self.dWnx = np.expand_dims(dtanh, axis=1) @ np.expand_dims(self.x, axis=1).T  # 计算 Wnx 的梯度
        self.dbnx = dtanh  # 计算 bnx 的梯度
        dr = dtanh * (self.Wnh @ self.hidden + self.bnh)  # 计算重置门 r 的梯度
        self.dWnh = np.expand_dims(dtanh, axis=1) * np.expand_dims(self.r, axis=1) @ np.expand_dims(self.hidden, axis=1).T  # 计算 Wnh 的梯度
        self.dbnh = dtanh * self.r  # 计算 bnh 的梯度

        # 计算 sigmoid 激活函数的反向传播
        dsigz = self.z_act.backward(dz)
        self.dWzx = np.expand_dims(dsigz, axis=1) @ np.expand_dims(self.x, axis=1).T  # 计算 Wzx 的梯度
        self.dbzx = dsigz  # 计算 bzx 的梯度
        self.dWzh = np.expand_dims(dsigz, axis=1) * np.expand_dims(self.hidden, axis=1).T  # 计算 Wzh 的梯度
        self.dbzh = dsigz  # 计算 bzh 的梯度

        # 计算 sigmoid 激活函数的反向传播（重置门 r）
        dsigr = self.r_act.backward(dr)
        self.dWrx = np.expand_dims(dsigr, axis=1) @ np.expand_dims(self.x, axis=1).T  # 计算 Wrx 的梯度
        self.dbrx = dsigr  # 计算 brx 的梯度
        self.dWrh = np.expand_dims(dsigr, axis=1) * np.expand_dims(self.hidden, axis=1).T  # 计算 Wrh 的梯度
        self.dbrh = dsigr  # 计算 brh 的梯度

        # 计算输入 x 的梯度
        dx = np.squeeze((np.expand_dims(dtanh, axis=1).T @ self.Wnx + np.expand_dims(dsigz, axis=1).T @ self.Wzx + np.expand_dims(dsigr, axis=1).T @ self.Wrx).T)
        # 计算前一时间步隐藏状态 h 的梯度
        dh_prev_t = delta * self.z + dtanh * self.r @ self.Wnh + dsigz @ self.Wzh + dsigr @ self.Wrh

        # 确保梯度的维度正确
        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t  # 返回输入 x 和前一时间步隐藏状态 h 的梯度
