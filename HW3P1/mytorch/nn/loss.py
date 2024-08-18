import numpy as np
import os

# 这个 Criterion 类作为损失函数的接口。所有的损失函数类都将继承自这个基类。
class Criterion(object):
    """
    损失函数的接口类。
    """

    def __init__(self):
        # 初始化存储logits、标签和损失的变量
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        # 当类的实例被调用时，自动调用 forward 方法
        return self.forward(x, y)

    def forward(self, x, y):
        # 前向传播方法，子类必须实现这个方法
        raise NotImplemented

    def derivative(self):
        # 计算损失函数对输入的导数，子类必须实现这个方法
        raise NotImplemented

# SoftmaxCrossEntropy 类实现了Softmax交叉熵损失函数
class SoftmaxCrossEntropy(Criterion):
    """
    Softmax 交叉熵损失函数
    """

    def __init__(self):
        # 调用父类 Criterion 的初始化方法
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        前向传播，计算损失值

        参数:
            x (np.array): 输入的logits，维度为 (batch size, 10)
            y (np.array): 真实标签，维度为 (batch size, 10)
        
        返回:
            out (np.array): 每个样本的损失值，维度为 (batch size, )
        """
        # 存储输入的 logits 和标签
        self.logits = x
        self.labels = y
        self.batch_size = self.labels.shape[0]

        # 计算每个类的指数
        exps = np.exp(self.logits)
        # 计算 softmax 输出
        self.softmax = exps / exps.sum(axis=1, keepdims=True)
        # 计算交叉熵损失
        self.loss = np.sum(np.multiply(self.labels, -np.log(self.softmax)), axis=1)

        return self.loss

    def backward(self):
        """
        反向传播，计算损失函数对logits的梯度

        返回:
            out (np.array): 损失对logits的导数，维度为 (batch size, 10)
        """
        # 计算 softmax 输出与真实标签之间的差值，即梯度
        self.gradient = self.softmax - self.labels

        return self.gradient
