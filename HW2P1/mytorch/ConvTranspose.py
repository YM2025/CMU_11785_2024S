import numpy as np
from resampling import *
from Conv1d import *
from Conv2d import *


class ConvTranspose1d():
    """
    实现一维转置卷积类。
    """
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        """
        初始化一维转置卷积层。
        :param in_channels: 输入通道数。
        :param out_channels: 输出通道数。
        :param kernel_size: 卷积核大小。
        :param upsampling_factor: 上采样因子。
        :param weight_init_fn: 权重初始化函数。
        :param bias_init_fn: 偏置初始化函数。
        """
        self.upsampling_factor = upsampling_factor  # 设置上采样因子

        # 初始化一维上采样和一维卷积实例，这里假设Upsample1d和Conv1d_stride1类已经实现
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)

    def forward(self, A):
        """
        前向传播过程。
        :param A: 输入数组，维度为(batch_size, in_channels, input_size)。
        :return: 输出数组，维度为(batch_size, out_channels, output_size)。
        """
        # 上采样输入
        A_upsampled = self.upsample1d.forward(A)

        # 应用步长为1的一维卷积
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        反向传播过程。
        :param dLdZ: 损失关于输出Z的梯度。
        :return: 损失关于输入A的梯度。
        """
        # 首先反向传播通过步长为1的一维卷积层
        delta_out = self.conv1d_stride1.backward(dLdZ)

        # 然后反向传播通过上采样层
        dLdA = self.upsample1d.backward(delta_out)

        return dLdA



class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        """
        二维转置卷积的初始化方法。
        :param in_channels: 输入的通道数。
        :param out_channels: 输出的通道数。
        :param kernel_size: 卷积核的大小，可以是一个整数或一个由两个整数构成的元组。
        :param upsampling_factor: 上采样因子，即输出相对于输入在空间维度上的放大倍数。
        :param weight_init_fn: 权重初始化函数，用于初始化卷积核的权重。
        :param bias_init_fn: 偏置初始化函数，用于初始化卷积核的偏置。
        """
        # 存储上采样因子
        self.upsampling_factor = upsampling_factor

        # 初始化二维卷积实例，这里Conv2d_stride1类应该实现了常规的二维卷积操作，其中步长被设置为1。
        # 此卷积用于在上采样后的数据上应用。
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)
        # 初始化上采样实例，Upsample2d类应该实现了二维数据的上采样操作，按照指定的上采样因子放大数据。
        self.upsample2d = Upsample2d(upsampling_factor)

    def forward(self, A):
        """
        对输入数据执行前向传播。
        :param A: 输入的numpy数组，其形状应该是(batch_size, in_channels, height, width)。
        :return: 输出的numpy数组，其形状是(batch_size, out_channels, new_height, new_width)，其中
                 new_height和new_width由上采样因子和卷积核大小共同决定。
        """
        # 首先对输入数据A执行上采样操作，得到上采样后的数据A_upsampled。
        A_upsampled = self.upsample2d.forward(A)

        # 然后在上采样后的数据A_upsampled上执行步长为1的二维卷积操作，得到最终的输出Z。
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        对损失关于输出Z的梯度执行反向传播。
        :param dLdZ: 损失关于输出Z的梯度，其形状应该与forward方法的输出Z相同。
        :return: 损失关于输入A的梯度dLdA，其形状与输入A相同。
        """
        # 首先对损失梯度dLdZ执行二维卷积的反向传播操作，得到经过卷积层反向传播后的梯度delta_out。
        delta_out = self.conv2d_stride1.backward(dLdZ)

        # 然后对delta_out执行上采样的反向传播操作，得到最终的损失关于输入A的梯度dLdA。
        dLdA = self.upsample2d.backward(delta_out)

        return dLdA
