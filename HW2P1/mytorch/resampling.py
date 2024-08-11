import numpy as np
import pdb  # 导入pdb模块


class Upsample1d():
    def __init__(self, upsampling_factor):
        """
        类初始化函数。
        
        参数:
            upsampling_factor (int): 上采样因子。
        """
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        上采样的前向传播。
        
        参数:
            A (np.array): 输入数组，形状为(batch_size, in_channels, input_width)。
        
        返回:
            Z (np.array): 上采样后的数组，形状为(batch_size, in_channels, output_width)。
        """
        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1
        
        # 初始化输出数组
        Z = np.zeros((batch_size, in_channels, output_width))
        
        # 使用切片操作填充非零值
        Z[:, :, ::self.upsampling_factor] = A # 从0开始，每隔upsampling_factor个元素赋值为A中的元素
    
        return Z

    def backward(self, dLdZ):
        """
        上采样的反向传播。
        
        参数:
            dLdZ (np.array): 损失函数关于上采样输出的梯度，形状为(batch_size, in_channels, output_width)。
        
        返回:
            dLdA (np.array): 损失函数关于上采样输入的梯度，形状为(batch_size, in_channels, input_width)。
        """
        
        # 从上采样的梯度中提取原始像素位置的梯度
        dLdA = dLdZ[:, :, ::self.upsampling_factor] # 从0开始，每隔upsampling_factor个元素提取dLdZ中的元素
        
        return dLdA



class Downsample1d():
    def __init__(self, downsampling_factor):
        """
        类的构造函数，初始化下采样因子。

        参数:
            downsampling_factor (int): 下采样因子。
        """
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        前向传播函数，实现下采样操作。
        
        参数:
            A (np.array): 输入数组，形状为(batch_size, in_channels, input_width)。
        
        返回:
            Z (np.array): 下采样后的数组，形状为(batch_size, in_channels, output_width)。
        """
        # 存储原始输入宽度，以便在反向传播时使用
        self.input_width = A.shape[2]
        
        # 使用切片操作进行下采样
        Z = A[:, :, ::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        反向传播函数，将梯度映射回下采样前的维度。
        
        参数:
            dLdZ (np.array): 关于下采样输出的损失梯度，形状为(batch_size, in_channels, output_width)。
        
        返回:
            dLdA (np.array): 映射回原始输入尺寸的损失梯度，形状为(batch_size, in_channels, input_width)。
        """
        batch_size, in_channels, _ = dLdZ.shape
        
        # 初始化一个全零的梯度数组，根据存储的原始输入宽度设置大小
        dLdA = np.zeros((batch_size, in_channels, self.input_width))
        
        # 将损失梯度映射回被下采样的位置
        dLdA[:, :, ::self.downsampling_factor] = dLdZ # 从0开始，每隔downsampling_factor个元素赋值为dLdZ中的元素
        
        return dLdA





class Upsample2d():
    def __init__(self, upsampling_factor):
        """
        初始化Upsample2d类的实例。
        
        参数:
        upsampling_factor (int): 上采样因子，即在每个像素之间插入的零的个数加一（在x和y方向上）。
        """
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        对输入的图像进行上采样（膨胀）。
        
        参数:
        A (numpy array): 输入的图像，形状为(N, C, H_in, W_in)。
        
        返回:
        Z (numpy array): 上采样后的图像，形状为(N, C, H_out, W_out)。
        """
        
        N, C, H_in, W_in = A.shape
        H_out = (H_in-1) * self.upsampling_factor + 1
        W_out = (W_in-1) * self.upsampling_factor + 1

        # 创建一个形状为(N, C, H_out, W_out)的新数组，初始值为零
        Z = np.zeros((N, C, H_out, W_out), dtype=A.dtype)
        
        # 使用切片操作填充非零值
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        对上采样后的梯度图进行下采样（池化），这是上采样的逆操作。
        
        参数:
        dLdZ (numpy array): 关于上采样后图像的损失梯度，形状为(N, C, H_out, W_out)。
        
        返回:
        dLdA (numpy array): 关于输入图像的损失梯度，形状为(N, C, H_in, W_in)。
        """

        # 从上采样的梯度中提取原始像素位置的梯度
        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA
    






class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        # 存储原始输入宽度，以便在反向传播时使用
        self.input_width, self.input_height = A.shape[2], A.shape[3]
        
        # 使用切片操作进行下采样
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        
        batch_size, in_channels, _, _ = dLdZ.shape
        
        # 初始化一个全零的梯度数组，根据存储的原始输入宽度设置大小
        dLdA = np.zeros((batch_size, in_channels, self.input_width, self.input_height))
        
        # 将损失梯度映射回被下采样的位置
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ # 从0开始，每隔downsampling_factor个元素赋值为dLdZ中的元素

        return dLdA
