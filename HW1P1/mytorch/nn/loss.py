import numpy as np


from mytorch.nn import Softmax


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
        self.N = np.shape(A)[0]  # TODO
        self.C = np.shape(A)[1]  # TODO
        
        se = (self.A - self.Y) ** 2
        sse = np.sum(se)
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        self.N = N

        Ones_C = np.ones(C, dtype='f')
        Ones_N = np.ones(N, dtype='f')

        # 计算softmax概率分布
        self.softmax = Softmax().forward(A) #调用softmax的前向传播
        # 计算交叉熵
        crossentropy = -Y * np.log(self.softmax + 1e-15) # 加上小数1e-15防止对数为负无穷
        sum_crossentropy = np.sum(np.dot(Ones_N.T, crossentropy))
        # 计算平均交叉熵损失
        L = sum_crossentropy / N

        return L

    def backward(self):
        # 计算交叉熵损失对A的梯度
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA