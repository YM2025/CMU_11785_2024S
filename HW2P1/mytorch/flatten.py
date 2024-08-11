import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A_shape = A.shape # 保存输入形状以便在反向传播时使用
        Z = A.reshape(A.shape[0], -1)  # A.reshape(batch_size, in_channels * in_width)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.A_shape)  # 将梯度重新塑形为原始输入形状

        return dLdA