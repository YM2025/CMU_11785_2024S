import torch

class Softmax:

    '''
    不要修改！Softmax类已经在Attention类的构造函数中初始化，可以直接使用。
    该类实现了在最后一个维度上的softmax操作。
    '''
    def forward(self, Z):

        # 存储输入Z的原始形状，以便后续reshape操作
        z_original_shape = Z.shape

        # N是批次大小乘以序列长度，C是输入的最后一个维度（词汇量的大小）
        self.N = Z.shape[0] * Z.shape[1]
        self.C = Z.shape[2]

        # 将Z reshape成二维矩阵，以便进行矩阵运算
        Z = Z.reshape(self.N, self.C)

        # 初始化单位矩阵用于进行softmax归一化
        Ones_C = torch.ones((self.C, 1))

        # 计算softmax归一化值，并将结果存储在self.A中
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        # 返回与原始Z形状一致的softmax结果
        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        # 存储dLdA的原始形状，以便后续reshape操作
        dLdA_original_shape = dLdA.shape

        # 将dLdA reshape成二维矩阵，以便进行矩阵运算
        dLdA = dLdA.reshape(self.N, self.C)

        # 初始化dLdZ，用于存储对Z的梯度
        dLdZ = torch.zeros((self.N, self.C))
        
        # 遍历每一个样本，计算每个样本的Jacobian矩阵
        for i in range(self.N):

            # 初始化Jacobian矩阵
            J = torch.zeros((self.C, self.C))

            # 计算Jacobian矩阵的每个元素
            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        # 对角线元素的梯度公式
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        # 非对角线元素的梯度公式
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            # 计算当前样本的dLdZ
            dLdZ[i, :] = dLdA[i, :] @ J

        # 返回与原始dLdA形状一致的dLdZ
        return dLdZ.reshape(dLdA_original_shape)


class Attention:
        
        def __init__(self, weights_keys, weights_queries, weights_values):

            """
            初始化Attention类的权重参数。
            输入维度为D，键和查询的维度为D_k，值的维度为D_v

            参数说明：
            -----------
            weights_keys (torch.tensor, dim = (D X D_k)): 键的权重矩阵
            weights_queries (torch.tensor, dim = (D X D_k)): 查询的权重矩阵
            weights_values (torch.tensor, dim = (D X D_v)): 值的权重矩阵
            """

            # 将键、查询和值的权重存储为类的参数
            self.W_k = weights_keys
            self.W_q = weights_queries
            self.W_v = weights_values
            
            # 存储D_k和D_v的维度信息
            self.Dk = weights_keys.shape[1]
            self.Dv = weights_values.shape[1]

            # 初始化Softmax实例，用于后续的归一化计算
            self.softmax = Softmax()
        
        def forward(self, X):

            """
            计算自注意力层的输出。
            该函数将存储键、查询、值，以及未归一化和归一化的注意力权重。
            输入为一个批次数据，而非单一序列，因此在操作时应注意维度转换。

            输入
            -----
            X (torch.tensor, dim = (B, T, D)): 输入的批次张量

            返回
            ------
            X_new (torch.tensor, dim = (B, T, D_v)): 输出的批次张量
            """

            # 存储输入X
            self.X = X
        
            # 计算查询、键和值的矩阵
            self.Q = X @ self.W_q
            self.K = X @ self.W_k
            self.V = X @ self.W_v 

            # 计算未归一化的注意力分数（logits）
            self.A_w = self.Q @ self.K.transpose(1, 2)


            # 创建加性因果注意力掩码并应用
            # 使用torch.triu生成上三角矩阵作为掩码
            mask_single = torch.triu(torch.ones(self.A_w.shape[1], self.A_w.shape[2]), diagonal=1)
            # 为批次中的每个样本应用掩码
            mask_batch = mask_single.unsqueeze(0).repeat(self.A_w.shape[0], 1, 1)
            # 将掩码中为1的地方填充为负无穷大，避免其影响计算
            mask_final = mask_batch.masked_fill(mask_batch == 1, float('-inf'))

            # 将掩码应用到注意力分数中
            self.A_w_attn_mask = self.A_w + mask_final


            # 归一化注意力分数
            self.A_sig = self.softmax.forward(self.A_w_attn_mask / torch.sqrt(torch.tensor(self.Dk)))

            # 计算注意力上下文
            X_new = self.A_sig @ self.V

            # 返回新计算的上下文向量
            return X_new
            
        def backward(self, dLdXnew):

            """
            通过自注意力层进行反向传播。
            该函数将存储关于键、查询、值和权重矩阵的导数。
            输入为一个批次数据，而非单一序列，因此在操作时应注意维度转换。

            输入
            -----
            dLdXnew (torch.tensor, dim = (B, T, D_v)): 关于注意力层输出的梯度

            返回
            ------
            dLdX (torch.tensor, dim = (B, T, D)): 关于注意力层输入的梯度
            """

            # 计算关于归一化注意力权重的梯度
            dLdA_sig = dLdXnew @ self.V.transpose(1, 2)
            # 计算关于未归一化注意力权重的梯度
            dLdA_w = (1 / torch.sqrt(torch.tensor(self.Dk))) * self.softmax.backward(dLdA_sig).transpose(1, 2)

            # 计算关于值、键、查询的梯度
            self.dLdV = self.A_sig.transpose(1, 2) @ dLdXnew
            self.dLdK = dLdA_w @ self.Q
            self.dLdQ = dLdA_w.transpose(1, 2) @ self.K

            # 计算关于权重矩阵的梯度（需要在批次维度上求和）
            self.dLdWq = torch.sum(self.X.transpose(1, 2) @ self.dLdQ, dim=0)
            self.dLdWv = torch.sum(self.X.transpose(1, 2) @ self.dLdV, dim=0)
            self.dLdWk = torch.sum(self.X.transpose(1, 2) @ self.dLdK, dim=0)

            # 计算关于输入X的梯度
            dLdX = self.dLdV @ self.W_v.T + self.dLdK @ self.W_k.T + self.dLdQ @ self.W_q.T

            # 返回关于输入X的梯度
            return dLdX
