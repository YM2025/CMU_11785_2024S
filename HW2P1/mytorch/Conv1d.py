# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *



class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels # 输入通道数
        self.out_channels = out_channels # 输出通道数
        self.kernel_size = kernel_size  # 卷积核大小

        # 权重和偏置的初始化
        # self.W的维度：(out_channels, in_channels, kernel_size)
        # out_channels 是输出通道数（与卷积核的个数有关）
        # in_channels 是输入通道数（与输入数据的通道数相对应）
        if weight_init_fn is None:
            # 使用正态分布初始化权重
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            # 使用自定义的初始化函数初始化权重
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        # 初始化权重和偏置的梯度
        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        输入参数:
            A : (batch_size, in_channels, input_length)
                        input_length: 输入张量的长度
                        in_channels: 输入张量的通道数
            
        返回参数:
            Z : (batch_size, out_channels, output_length)
        """
        
        self.A = A # 保存输入张量，以便在反向传播时使用
        batch_size, _, input_length = A.shape
        output_length = input_length - self.kernel_size + 1

        # 初始化输出张量
        Z = np.zeros((batch_size, self.out_channels, output_length))

        # 计算卷积，
        # 方法1
        for i in range(output_length):
            # np.tensordot()：这个函数计算沿着指定轴的张量点积。
            # 在这里，它执行了被切片的输入张量 A[:, :, i:i+self.kernel_size] 和卷积核 self.W 的点积。
            # axes 参数 ([1, 2], [1, 2]) 表示点积沿着 A 的第2、3维度（in_channels, input_length），以及 W 的第2、3维度（in_channels, kernel_size）进行。
            Z[:, :, i] = np.tensordot(A[:, :, i:i+self.kernel_size], self.W, axes=([1, 2], [1, 2])) + self.b 
        
        # 方法2
        for n in range(Z.shape[0]): # 遍历batchsize
            for c in range(Z.shape[1]): # 遍历输出通道数
                for w in range(Z.shape[2]): # 遍历output_length
                    Z[n, c, w] = np.sum(A[n, :, w:w+self.kernel_size] * self.W[c,:,:]) + self.b[c]

        return Z


    def backward(self, dLdZ):
        """ 参数
                dLdZ :  (batch_size, out_channels, output_length)
                   A :  (batch_size, in_channels, input_length)
                dLdW :  (out_channels, in_channels, kernel_size)
            返回:
                dLdA :  (batch_size, in_channels, input_length)"""
            
        _, _, output_length = dLdZ.shape

        # （1）计算dLdW
        # 方法1
        for i in range(self.kernel_size):
            # axes参数([0, 2], [0, 2])表示点积沿着dLdZ和A的第一个和第三个维度进行。
            self.dLdW[:, :, i] = np.tensordot(dLdZ, self.A[:, :, i:i+output_length], axes=([0, 2], [0, 2]))
        
        # 方法2
        for o in range(self.out_channels): #遍历卷积核个数
            for i in range(self.in_channels): #遍历卷积核的通道数（与输入数据的通道数相同）
                for k in range(self.kernel_size): #遍历卷积核长度
                    self.dLdW[o, i, k] = np.sum(self.A[:, i, k:k+output_length] * dLdZ[:, o, :])


        # （2）计算dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2))


        # # （3）计算dLdA 
        # # 方法1
        # dLdA = np.zeros_like(self.A)
        # for i in range(output_length): #遍历输出特征图的每一个元素位置。
        #     dLdA[:, :, i:i+self.kernel_size] += np.tensordot(dLdZ[:, :, i], self.W, axes=([1], [0]))
        
        # # # 方法2
        # dLdA = np.zeros(self.A.shape) 
        # flipped_W = np.flip(self.W, axis=2) #将卷积核 W 在最后一个维度上（即宽度方向）进行翻转
        # # 在宽度方向上前、后各填充 self.kernel_size - 1 个0，保持其他维度不变。
        # # 填充操作是为了确保反卷积操作能够覆盖输入 A 的所有位置。卷积核大小为 self.kernel_size，
        # # 所以需要在 dL/dZ 的两边进行适当的填充。
        # padded_dLdZ = np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1, self.kernel_size-1)), 'constant') 
        # for n in range(dLdA.shape[0]): # 遍历输入数据的【batchsize】
        #     for c in range(dLdA.shape[1]): # 遍历输入数据的【通道数】（与卷积核的通道数相同）
        #         for w in range(dLdA.shape[2]): # 遍历输入数据的【宽度】（即input length）
        #             dLdA[n,c,w] = dLdA[n,c,w] + np.sum(padded_dLdZ[n, :, w:w+self.kernel_size] * flipped_W[:,c,:])

        # 方法3（本方法对应讲义中【图14】的计算过程）
        dLdA = np.zeros(self.A.shape)
        dLdZ_pad = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), mode='constant', constant_values=(0, 0))
        W_flipped = np.flip(self.W, axis=2)

        for i in range(dLdA.shape[2]): # 遍历输入数据的【宽度】（即input length）
            # 取出 dLdZ_pad 中从第 i 个位置开始，宽度为 self.kernel_size 的片段，作为一个局部区域进行计算
            section = dLdZ_pad[:, :, i:i+self.kernel_size]
            # 对局部区域 section 与翻转后的卷积核 W_flipped 进行张量点积，计算这个局部区域的梯度
            # axes=([1,2], [0,2]) 表示将 section 的第 1, 2 维与 W_flipped 的第 0, 2 维进行点积
            result = np.tensordot(section, W_flipped, axes=([1,2], [0,2]))
            # 将计算得到的梯度 result 存储到 dLdA 的对应位置
            dLdA[:,:,i] = result

        return dLdA




class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # 初始化步长 
        self.stride = stride

        # 初始化Conv1d_stride1实例和Downsample1d实例
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) 
        self.downsample1d = Downsample1d(stride)  

    def forward(self, A):
        """
        参数:
            A (np.array): (批量大小, 输入通道数, 输入尺寸)
        返回:
            Z (np.array): (批量大小, 输出通道数, 输出尺寸)
        """
        # 调用步长为1的1D卷积层
        con_A = self.conv1d_stride1.forward(A)
        
        # 执行下采样操作
        Z = self.downsample1d.forward(con_A)

        return Z

    def backward(self, dLdZ):
        """
        参数:
            dLdZ (np.array): (批量大小, 输出通道数, 输出尺寸)
        返回:
            dLdA (np.array): (批量大小, 输入通道数, 输入尺寸)
        """
        # 调用下采样层的反向传播方法
        down_dLdZ = self.downsample1d.backward(dLdZ)

        # 调用步长为1的1D卷积层的反向传播方法
        dLdA = self.conv1d_stride1.backward(down_dLdZ)

        return dLdA
