import sys
import numpy as np
import os

sys.path.append("mytorch")
from Conv2d import Conv2d_padding


from activation import *
from batchnorm2d import *



class ConvBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # 初始化一个卷积块，包括卷积层和批归一化层
        self.layers = [
            Conv2d_padding(in_channels, out_channels, kernel_size, stride, padding, bias_init_fn=None),
            BatchNorm2d(out_channels),
        ]

    def forward(self, A):
        # 前向传播，依次通过卷积层和批归一化层
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, grad):
        # 反向传播，依次逆序通过批归一化层和卷积层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class IdentityBlock(object):
    def forward(self, A):
        # 恒等块的前向传播，直接返回输入
        return A
	
    def backward(self, grad):
        # 恒等块的反向传播，直接返回梯度
        return grad


class ResBlock(object):
    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
        # 初始化残差块，包括卷积块、激活函数和残差连接
        self.convolution_layers = [
			ConvBlock(in_channels, out_channels, filter_size, stride, padding),
            ReLU(),
            ConvBlock(out_channels, out_channels, 1, 1, 0),
        ]
        self.final_activation = ReLU()  


        # 决定残差连接是使用卷积块还是恒等块
        # 当步幅（stride）不为1时，卷积层的输出大小会发生变化。因此，如果步幅不为1，残差连接就不能直接使用输入数据，需要通过卷积调整输入的尺寸。
        # 当输入通道数（in_channels）和输出通道数（out_channels）不相同时，输入数据与输出数据的形状不匹配，需要通过卷积调整通道数。
        # 当滤波器大小（filter_size）不为1时，意味着卷积核尺寸变大。一般情况下，这个条件会影响到卷积操作的范围.
        # 当填充（padding）不为0时，输入数据的尺寸会在卷积操作中保持不变或增大。因此，这个条件可能影响输出尺寸.
        
        # 在很多实现中，常见的残差块在判断是否需要卷积块来处理残差连接时，重点是步幅和通道数是否匹配。
        # 这两个条件（stride != 1 或 in_channels != out_channels）已经可以覆盖绝大多数情况下残差连接需要调整的场景。
        # 在S24版测试用例中，只使用stride != 1 or in_channels != out_channels也能通过。
        if stride != 1 or in_channels != out_channels or filter_size != 1 or padding != 0:
            # 当输入和输出维度不匹配时，使用卷积块作为残差连接
            self.residual_connection = ConvBlock(in_channels, out_channels, filter_size, stride, padding)
        else:
            # 当输入和输出维度匹配时，使用恒等块作为残差连接
            self.residual_connection = IdentityBlock()


    def forward(self, A):
        # 前向传播，通过卷积块和激活函数，并加上残差连接的输出
        Z = A
        for layer in self.convolution_layers:
            Z = layer.forward(Z)

        # 通过残差连接获取残差输出
        residual_out = self.residual_connection.forward(A)
        # 将卷积块的输出与残差连接的输出相加，并通过最终的ReLU激活函数
        return self.final_activation.forward(Z + residual_out)

    def backward(self, grad):
        # 残差块的反向传播，首先通过最终的ReLU激活函数计算梯度
        grad = self.final_activation.backward(grad)

        # 计算残差连接的梯度
        residual_grad = self.residual_connection.backward(grad)

        # 依次通过卷积块计算梯度
        for layer in reversed(self.convolution_layers):
            grad = layer.backward(grad)

        # 将卷积块的梯度与残差连接的梯度相加，得到最终的梯度
        return grad + residual_grad
