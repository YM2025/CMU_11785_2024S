import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        """
        初始化一个最大池化层。
        :param kernel: 池化核的大小，此处假设核为正方形。
        """
        self.kernel = kernel

    def forward(self, A):
        """
        前向传播函数：执行最大池化操作。
        :param A: 输入数据，形状为(batch_size, in_channels, input_width, input_height)。
        :return: 池化后的结果，形状为(batch_size, in_channels, output_width, output_height)。
        """
        self.A = A
        self.batch_size, self.in_channels, self.input_width, self.input_height = A.shape
        self.out_width = self.input_width - self.kernel + 1
        self.out_height = self.input_height - self.kernel + 1

        # 初始化输出矩阵
        Z = np.zeros((self.batch_size, self.in_channels, self.out_width, self.out_height))
        
        # 存储最大值位置的索引
        self.max_indices = np.zeros((self.batch_size, self.in_channels, self.out_width, self.out_height, 2), dtype=int)

        for n in range(self.batch_size):
            for c in range(self.in_channels):
                for i in range(self.out_width):
                    for j in range(self.out_height):
                        window = A[n, c, i:i+self.kernel, j:j+self.kernel]
                        max_val = np.max(window)
                        Z[n, c, i, j] = max_val
                        # 找到最大值的位置并存储
                        max_pos = np.unravel_index(np.argmax(window, axis=None), window.shape)
                        self.max_indices[n, c, i, j] = (i + max_pos[0], j + max_pos[1])

        return Z

    def backward(self, dLdZ):
        """
        反向传播函数：根据最大值的位置将梯度回传。
        :param dLdZ: 损失关于池化层输出的梯度，形状为(batch_size, in_channels, output_width, output_height)。
        :return: 损失关于池化层输入的梯度，形状与输入A相同。
        """
        dLdA = np.zeros_like(self.A)

        for n in range(self.batch_size):
            for c in range(self.in_channels):
                for i in range(self.out_width):
                    for j in range(self.out_height):
                        # 获取最大值位置的索引
                        (max_i, max_j) = self.max_indices[n, c, i, j]
                        # 只有最大值位置的梯度非零
                        dLdA[n, c, max_i, max_j] += dLdZ[n, c, i, j]

        return dLdA



class MeanPool2d_stride1:
    """
    前向传播 (forward 方法)
    1 获取输入尺寸: 从输入A中获取批量大小（batch_size）、通道数（in_channels）、输入宽度（input_width）和输入高度（input_height）。
    2 计算输出尺寸: 根据内核大小（kernel）和步长（stride=1）计算输出宽度（output_width）和输出高度（output_height）。
    3 初始化输出张量: 根据计算出的输出尺寸初始化输出张量Z。
    4 执行均值池化: 对于每个位置，计算覆盖的区域内元素的平均值，并将这个平均值赋值给输出张量的相应位置。
    
    反向传播 (backward 方法)
    1 获取梯度尺寸: 从dLdZ中获取批量大小、通道数、输出宽度和输出高度。
    2 初始化输入梯度张量: 初始化一个与输入A同样尺寸的张量dLdA，用于存放传回输入的梯度。
    3 计算输入梯度: 对于输出的每个梯度值，将其等分散布到对应的输入区域内的每个元素上。
    """
    def __init__(self, kernel):
        # 内核大小
        self.kernel = kernel

    def forward(self, A):
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        
        # 初始化输出张量
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        
        # 执行均值池化
        for i in range(output_width):
            for j in range(output_height):
                patch = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.mean(patch, axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        dLdA = np.zeros((batch_size, out_channels, output_width + self.kernel - 1, output_height + self.kernel - 1))
        
        # 计算输入的梯度
        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j][:, :, None, None] / (self.kernel * self.kernel)
        
        return dLdA



class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # 假设存在MaxPool2d_stride1和downsample2d，并进行初始化
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        使用步长为1的最大池化，然后下采样到期望的步长
        """
        # 首先进行步长为1的最大池化
        Z = self.maxpool2d_stride1.forward(A)
        
        # 根据stride进行下采样
        Z = self.downsample2d.forward(Z)
        
        return Z

    def backward(self, dLdZ):
        """
        反向传播同样需要考虑下采样和最大池化的梯度传递
        """
        # 首先根据stride对梯度进行上采样
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        
        # 然后将上采样的梯度通过步长为1的最大池化的反向传播
        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)
        
        return dLdA



class MeanPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # 假设MeanPool2d_stride1和Downsample2d已经定义，并进行初始化
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        使用步长为1的均值池化，然后下采样到期望的步长
        """
        # 首先进行步长为1的均值池化
        Z = self.meanpool2d_stride1.forward(A)
        
        # 根据stride进行下采样
        Z = self.downsample2d.forward(Z)
        
        return Z

    def backward(self, dLdZ):
        """
        反向传播同样需要考虑下采样和均值池化的梯度传递
        """
        # 首先根据stride对梯度进行上采样
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        
        # 然后将上采样的梯度通过步长为1的均值池化的反向传播
        dLdA = self.meanpool2d_stride1.backward(dLdZ_upsampled)
        
        return dLdA
