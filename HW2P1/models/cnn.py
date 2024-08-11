# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys



class CNN(object):
    """
    实现一个简单的卷积神经网络模型。
    你需要在下面的 get_cnn_model 函数中指定详细的模型架构，其架构应与第3.3节图3相同。
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        初始化CNN模型的参数。
        
        参数:
        input_width           : int     : 第一个卷积层输入的宽度
        num_input_channels    : int     : 输入层的通道数
        num_channels          : [int]   : 每个卷积层的输出通道数列表
        kernel_sizes          : [int]   : 每个卷积层的核宽度列表
        strides               : [int]   : 每个卷积层的步长列表
        num_linear_neurons    : int     : 线性层的神经元数
        activations           : [obj]   : 每个卷积层对应的激活函数对象列表
        conv_weight_init_fn   : fn      : 初始化卷积层权重的函数
        bias_init_fn          : fn      : 初始化所有卷积层和线性层偏置的函数
        linear_weight_init_fn : fn      : 初始化线性层权重的函数
        criterion             : obj     : 使用的损失函数对象（例如SoftMaxCrossEntropy）
        lr                    : float   : 学习率

        注意: activations, num_channels, kernel_sizes, strides 的长度必须一致。
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------
        
        # 初始化卷积层
        self.convolutional_layers = [Conv1d(num_input_channels, num_channels[0], kernel_sizes[0], strides[0]),
                                     Conv1d(num_channels[0], num_channels[1], kernel_sizes[1], strides[1]),
                                     Conv1d(num_channels[1], num_channels[2], kernel_sizes[2], strides[2])]
        self.flatten = Flatten()
        
        # 计算每个卷积层的输出尺寸
        conv1_out = (input_width-kernel_sizes[0])/strides[0]+1 
        conv2_out = (conv1_out-kernel_sizes[1])/strides[1]+1
        conv3_out = (conv2_out-kernel_sizes[2])/strides[2]+1
        self.linear_layer = Linear(int(conv3_out*num_channels[2]), num_linear_neurons)
        
        # 将所有层组合到一个列表中
        self.layers = [self.convolutional_layers[0], 
                       activations[0], 
                       self.convolutional_layers[1], 
                       activations[1], 
                       self.convolutional_layers[2], 
                       activations[2], 
                       self.flatten, 
                       self.linear_layer
                      ]

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        # Your code goes here -->
        # Iterate through each layer
        # <---------------------

        # Save output (necessary for error and loss)
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)

        self.Z = Z

        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        # Your code goes here -->
        # Iterate through each layer in reverse order
        # <---------------------
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grads(self):
        # Do not modify this method
        # 清零所有梯度
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):
        # Do not modify this method
        # # 更新权重
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
