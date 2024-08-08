import numpy as np

class SGD:
    def __init__(self, model, lr=0.1, momentum=0):
        """
        初始化SGD优化器。
        :param model: 包含需要优化的参数的模型。
        :param lr: 学习率，默认为0.1。
        :param momentum: 动量系数，默认为0，如果不为0，则应用动量优化。
        """
        # 筛选出模型中所有含有权重W的层，存入列表self.l
        # self.l = list(filter(lambda x: hasattr(x, 'W'), model.layers))
        self.l = model.layers # HW1P1测试用例可直接使用模型所有层
        # 获取模型层数量
        self.L = len(self.l)
        # 设置学习率本
        self.lr = lr
        # 设置动量系数
        self.mu = momentum
        # 初始化权重和偏置的动量变量为0，形状与对应权重相同
        self.v_W = [np.zeros_like(layer.W) for layer in self.l]
        self.v_b = [np.zeros_like(layer.b) for layer in self.l]

    def step(self):
        """
        执行一步参数更新。
        """
        # 遍历每一层
        for i in range(self.L):
            if self.mu == 0:  # 如果动量系数为0，执行标准的SGD更新
                # 直接用当前层的梯度和学习率更新权重和偏置
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb
            else:
                # 更新动量
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb
                # 使用更新后的动量变量更新权重和偏置
                self.l[i].W -= self.lr * self.v_W[i]
                self.l[i].b -= self.lr * self.v_b[i]

