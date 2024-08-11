# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import Flatten
from Conv1d import Conv1d
from linear import Linear
from activation import ReLU
from loss import CrossEntropyLoss
import numpy as np
import os
import sys



class CNN_SimpleScanningMLP():
    def __init__(self):
        # 第一层卷积层，模拟MLP中对输入数据的首次处理。
        # in_channels=24：作业讲义中提到总共128个数据点，每个数据有24个特征，即输入数据通道数是24。注意这里没有batchsize。
        # out_channels=8： 表示此卷积层生成8个特征映射，类似于MLP中第一层有8个神经元
        # kernel_size=8：1D卷积每次处理8个数据点。
        # stride=4： 每次滑动4个数据点距离。
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)

        # 第二层卷积层，使用1x1卷积核实现类似全连接层的效果。
        # out_channels=16，相当于MLP中第二层的16个神经元。
        # kernel_size=1 和 stride=1 保证这一层完全连接到前一层的所有输出。
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)

        # 第三层卷积层，同样使用1x1卷积核。
        # out_channels=4，对应MLP中最后一层的4个输出神经元。
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        # 经过3层1D卷积后，128*24的数据，最终输出维度是 31*4，其中31是128个数据点进行窗口为8步长为4的滑动结果。
        
        # 构建激活层和最后的扁平化层
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()  # 将多维输出扁平化成单一维度，模拟MLP的行为
        ]
        
    def init_weights(self, weights):
        # 初始化权重，将MLP的权重转置并赋给相应的卷积层
        # 注意，这里需要将MLP的权重矩阵转置，因为在MLP中权重矩阵的维度是(out_features, in_features)
        w1, w2, w3 = weights
        w1 = np.transpose(w1).reshape((self.conv1.conv1d_stride1.out_channels, self.conv1.conv1d_stride1.kernel_size, self.conv1.conv1d_stride1.in_channels))
        w2 = np.transpose(w2).reshape((self.conv2.conv1d_stride1.out_channels, self.conv2.conv1d_stride1.kernel_size, self.conv2.conv1d_stride1.in_channels))
        w3 = np.transpose(w3).reshape((self.conv3.conv1d_stride1.out_channels, self.conv3.conv1d_stride1.kernel_size, self.conv3.conv1d_stride1.in_channels))
        self.conv1.conv1d_stride1.W = np.transpose(w1, (0,2,1))
        self.conv2.conv1d_stride1.W = np.transpose(w2, (0,2,1))
        self.conv3.conv1d_stride1.W = np.transpose(w3, (0,2,1))
        
    def forward(self, A):
        """
        执行模型的前向传播。
        参数:
            A (np.array): 输入数组，形状为 (batch size, in channel, in width)
        返回:
            Z (np.array): 输出数组，形状为 (batch size, out channel, out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        执行模型的后向传播。
        参数:
            dLdZ (np.array): 损失关于输出的梯度
        返回:
            dLdA (np.array): 损失关于输入的梯度
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA



class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        w1, w2, w3 = weights
        
        w1 = np.reshape(w1[:48,:2].T, (2,2,24)) #2,2,24
        w2 = np.reshape(w2[:4,:8].T, (8,2,2))   #8,2,2
        w3 = np.reshape(w3[:16,:4].T, (4,2,8))  #4,2,8

        w1 = np.transpose(w1, (0,2,1))  #2,24,2
        w2 = np.transpose(w2, (0,2,1))  #8,2,2
        w3 = np.transpose(w3, (0,2,1))  #4,8,2
        self.conv1.conv1d_stride1.W = w1
        self.conv2.conv1d_stride1.W = w2
        self.conv3.conv1d_stride1.W = w3

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
