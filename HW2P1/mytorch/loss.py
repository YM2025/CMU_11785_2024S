import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (A - Y) ** 2  # TODO
        sse = np.sum(se)  # TODO
        mse = sse / (self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA




class CrossEntropyLoss:

    def softmax(self, x):
        """
        计算softmax概率分布。
        :param x: 输入数组。
        :return: softmax概率分布。
        """
        # 防止指数运算溢出，通过减去每一行的最大值来进行数值稳定化
        x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, A, Y):
        """
        计算交叉熵损失。
        :param A: 模型的输出，形状为(N, C)，其中N是样本数量，C是类别数。
        :param Y: 真实标签的独热编码，形状与A相同。
        :return: 交叉熵损失的标量值。
        """
        self.A = A
        self.Y = Y
        N, C = A.shape  # N是样本数量，C是类别数

        # 计算softmax概率分布
        self.softmax = self.softmax(A)
        # 计算交叉熵
        crossentropy = -np.sum(Y * np.log(self.softmax + 1e-15))  # 加上小数1e-15防止对数为负无穷
        # 计算平均交叉熵损失
        L = crossentropy / N

        return L

    def backward(self):
        """
        计算损失关于模型输出的梯度。
        :return: 损失关于A的梯度。
        """
        N = self.Y.shape[0]  # N是样本数量
        # 计算交叉熵损失对A的梯度
        dLdA = (self.softmax - self.Y) / N

        return dLdA
