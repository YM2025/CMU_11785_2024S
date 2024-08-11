import numpy as np

np.random.seed(11785)  

class Dropout2d(object):
    def __init__(self, p=0.5):
        # 初始化dropout概率p，默认为0.5
        self.p = p
        self.mask = None  # 初始化掩码，用于在训练过程中随机关闭部分神经元

    def __call__(self, *args, **kwargs):
        # 使得类实例可以像函数一样被调用，内部调用forward方法
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        前向传播方法：
        参数:
          x (np.array): 输入数据，其形状为(batch_size, in_channel, input_width, input_height)
          eval (boolean): 表示模型是否处于评估模式
        返回:
          和输入形状相同的np.array
        """
        if eval:
            # 如果是评估模式，直接返回输入数据x，不进行任何dropout操作
            return x
        else:
            # 如果是训练模式，进行dropout操作
            self.mask = np.random.binomial(1, 1-self.p, size=(x.shape[0], x.shape[1], 1, 1))
            # 生成一个随机掩码，掩码中的元素非0即1，1的概率为1-p
            return x * self.mask * 1/(1-self.p)
            # 应用掩码，并通过1/(1-p)缩放以保持激活值的总量不变

    def backward(self, delta):
        """
        反向传播方法：
        参数:
          delta (np.array): 上游传来的梯度，其形状为(batch_size, in_channel, input_width, input_height)
        返回:
          和输入梯度形状相同的np.array
        """
        # 训练模式下，进行梯度传播
        return delta * self.mask * 1/(1-self.p)
        # 通过已保存的掩码对梯度进行调整，再通过1/(1-p)缩放
