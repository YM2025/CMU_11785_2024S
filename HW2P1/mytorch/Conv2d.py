import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # 构造函数初始化2D卷积层
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = kernel_size  # 卷积核尺寸

        # 权重初始化
        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        # 偏置初始化
        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)  # 用于存储权重的梯度
        self.dLdb = np.zeros(self.b.shape)  # 用于存储偏置的梯度

    def forward(self, A):
        """
        前向传播函数
        参数:
            A (np.array): 输入数据, 形状为(batch_size, in_channels, input_height, input_width)
        返回:
            Z (np.array): 卷积操作后的输出, 形状为(batch_size, out_channels, output_height, output_width)
        """
        batch_size, _, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        
        # 初始化输出Z
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # 进行卷积操作
        for i in range(output_height):
            for j in range(output_width):
                # 提取输入数据的局部区域
                patch = A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                # 在局部区域和卷积核之间进行元素乘法操作，并对结果求和
                for k in range(self.out_channels):
                    Z[:, k, i, j] = np.sum(patch * self.W[k, :, :, :], axis=(1, 2, 3)) # axis=(1, 2, 3)表示沿着(通道数，高度，宽度)方向求和
        
        # 添加偏置
        Z += self.b.reshape((1, -1, 1, 1))

        self.A = A  # 保存输入数据用于反向传播
        return Z

    def backward(self, dLdZ):
        """
        反向传播函数
        参数:
            dLdZ (np.array): 损失对输出Z的梯度, 形状为(batch_size, out_channels, output_height, output_width)
        返回:
            dLdA (np.array): 损失对输入A的梯度, 形状为(batch_size, in_channels, input_height, input_width)
        """
        # 初始化梯度数组
        self.dLdW.fill(0)
        self.dLdb.fill(0)
        batch_size, _, input_height, input_width = self.A.shape
        _, _, output_height, output_width = dLdZ.shape
        
        # 计算dLdA
        dLdA = np.zeros_like(self.A, dtype=np.float64)
        for i in range(output_height):
            for j in range(output_width):
                for k in range(self.out_channels):
                    dLdA[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += dLdZ[:, k, i, j][:, None, None, None] * self.W[k, :, :, :]

        # 计算dLdW
        for k in range(self.out_channels):
            for i in range(output_height):
                for j in range(output_width):
                    self.dLdW[k, :, :, :] += np.sum(dLdZ[:, k, i, j][:, None, None, None] * self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size], axis=0)

        # 计算dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        return dLdA 



class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # 初始化2维卷积层，可以接受任意步长
        self.stride = stride

        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  
        self.downsample2d = Downsample2d(stride)  

    def forward(self, A):
        """
        参数:
            A (np.array): 输入数据, 形状为(batch_size, in_channels, input_height, input_width)
        返回:
            Z (np.array): 卷积操作后的输出, 形状为(batch_size, out_channels, output_height, output_width)
        """
        # 首先执行步长为1的2维卷积
        conv_A = self.conv2d_stride1.forward(A)
        
        # 然后进行下采样以实现所需的步长
        Z = self.downsample2d.forward(conv_A)
        
        return Z

    def backward(self, dLdZ):
        """
        参数:
            dLdZ (np.array): 对输出Z的损失梯度, 形状为(batch_size, out_channels, output_height, output_width)
        返回:
            dLdA (np.array): 对输入A的损失梯度, 形状为(batch_size, in_channels, input_height, input_width)
        """
        
        # 这个顺序很重要，首先处理下采样层的梯度，然后再处理卷积层的梯度
        
        # 首先处理下采样层的梯度
        down_dLdZ = self.downsample2d.backward(dLdZ)

        # 将梯度传递给步长为1的卷积层
        dLdA = self.conv2d_stride1.backward(down_dLdZ)  
        
        return dLdA




class Conv2d_padding():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # 初始化2维卷积层，可以接受任意步长
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  
        self.downsample2d = Downsample2d(stride)  

    def forward(self, A):
        """
        参数:
            A (np.array): 输入数据, 形状为(batch_size, in_channels, input_height, input_width)
        返回:
            Z (np.array): 卷积操作后的输出, 形状为(batch_size, out_channels, output_height, output_width)
        """

        # 在输入张量 A 的高度和宽度两个维度上进行零填充，填充的宽度为 self.pad
        # 假设 A 的形状为 (10, 3, 32, 32)，self.pad = 1，那么执行这行代码后，
        # A 的形状将变为 (10, 3, 34, 34)，即在每个输入图像的四周各添加了一圈零填充。
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')

        # 首先执行步长为1的2维卷积
        conv_A = self.conv2d_stride1.forward(A)
        
        # 然后进行下采样以实现所需的步长
        Z = self.downsample2d.forward(conv_A)
        
        return Z

    def backward(self, dLdZ):
        """
        参数:
            dLdZ (np.array): 对输出Z的损失梯度, 形状为(batch_size, out_channels, output_height, output_width)
        返回:
            dLdA (np.array): 对输入A的损失梯度, 形状为(batch_size, in_channels, input_height, input_width)
        """
        
        # 首先处理下采样层的梯度
        down_dLdZ = self.downsample2d.backward(dLdZ)

        # 将梯度传递给步长为1的卷积层
        dLdA = self.conv2d_stride1.backward(down_dLdZ)  
        
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]
        
        return dLdA


