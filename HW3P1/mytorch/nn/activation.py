# 请不要导入任何额外的第三方外部库，因为这些库在AutoLab中不可用，也不需要（或不允许）使用。
import numpy as np

class Sigmoid:
    """
    Sigmoid 激活函数类
    """

    def forward(self, Z):
        """
        前向传播，计算 Sigmoid 函数的输出。

        参数:
        -----
        Z : 输入数据 (可以是标量、向量或矩阵)

        返回:
        -----
        Sigmoid 函数的输出值
        """
        self.A = Z
        self.npVal = np.exp(-self.A)
        return 1 / (1 + self.npVal)

    def backward(self, dLdA):
        """
        反向传播，计算 Sigmoid 函数的导数。

        参数:
        -----
        dLdA : 上一层传递过来的损失梯度

        返回:
        -----
        当前层的损失梯度，用于反向传播更新参数
        """
        dAdZ = self.npVal / (1 + self.npVal) ** 2
        return dAdZ * dLdA

class Tanh:
    """
    修改过的 Tanh 激活函数类，用于反向传播中的 BPTT（Backpropagation Through Time）。
    Tanh(x) 的结果需要存储在其他地方，否则我们需要在每个时间步中为每个单元存储多个时间步的结果，
    这可能被认为是不好的设计。

    在计算导数时，我们可以传入存储的隐藏状态并计算该状态的导数，而不是计算当前存储状态的导数，
    因为当前存储状态可能会有所不同。
    """

    def forward(self, Z):
        """
        前向传播，计算 Tanh 函数的输出。

        参数:
        -----
        Z : 输入数据 (可以是标量、向量或矩阵)

        返回:
        -----
        Tanh 函数的输出值
        """
        self.A = Z
        self.tanhVal = np.tanh(self.A)
        return self.tanhVal

    def backward(self, dLdA, state=None):
        """
        反向传播，计算 Tanh 函数的导数。

        参数:
        -----
        dLdA : 上一层传递过来的损失梯度
        state : 可选参数，存储的隐藏状态，如果提供了该状态，则使用它计算导数

        返回:
        -----
        当前层的损失梯度，用于反向传播更新参数
        """
        if state is not None:
            dAdZ = 1 - state * state
            return dAdZ * dLdA
        else:
            dAdZ = 1 - self.tanhVal * self.tanhVal
            return dAdZ * dLdA
