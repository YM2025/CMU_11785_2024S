import numpy as np

class CTC(object):

    def __init__(self, BLANK=0):
        """
        初始化实例变量

        参数:
        ------
        BLANK (int, 可选): 空白标签的索引。默认值为0。
        """
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """用空白符扩展目标序列。

        输入:
        -----
        target: (np.array, 维度 = (target_len,))
                目标输出，包含目标音素的索引
        示例: [1,4,4,7]

        返回:
        ------
        extSymbols: (np.array, 维度 = (2 * target_len + 1,))
                    扩展了空白符的目标序列
        示例: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, 维度 = (2 * target_len + 1,))
                    跳跃连接
        示例: [0,0,0,1,0,0,0,1,0]
        """

        # 初始化扩展符号序列，首先添加一个空白符
        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)  # 添加目标符号
            extended_symbols.append(self.BLANK)  # 添加空白符

        N = len(extended_symbols)  # 扩展后的序列长度
        skip_connect = np.zeros(N)  # 初始化跳跃连接数组，默认值为0

        for i, sy in enumerate(target):
            if i != 0:
                # 如果当前符号与前一个符号不同，设置跳跃连接为1
                if target[i] != target[i-1]:
                    skip_connect[2*i+1] = 1

        extended_symbols = np.array(extended_symbols).reshape((N,))  # 转换为numpy数组
        skip_connect = np.array(skip_connect).reshape((N,))  # 转换为numpy数组

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """计算前向概率。

        输入:
        -----
        logits: (np.array, 维度 = (input_len, len(Symbols)))
                预测的（对数）概率

                在时间步t获取符号i的对数概率:
                p(t,s(i)) = logits[t, extended_symbols[i]]

        extSymbols: (np.array, 维度 = (2 * target_len + 1,))
                    扩展了空白符的标签序列

        skipConnect: (np.array, 维度 = (2 * target_len + 1,))
                    跳跃连接

        返回:
        ------
        alpha: (np.array, 维度 = (input_len, 2 * target_len + 1))
                前向概率
        """

        S, T = len(extended_symbols), len(logits)  # S为扩展后的序列长度，T为输入序列长度
        alpha = np.zeros(shape=(T, S))  # 初始化前向概率矩阵

        # 初始时间步的前向概率
        alpha[0, 0] = logits[0, extended_symbols[0]]  # 第一个符号的前向概率
        alpha[0, 1] = logits[0, extended_symbols[1]]  # 第二个符号的前向概率

        for t in range(1, T):
            # 计算时间步t的第一个符号的前向概率
            alpha[t, 0] = alpha[t-1, 0] * logits[t, extended_symbols[0]]
            for r in range(1, S):
                # 当前符号的前向概率等于前一个符号和当前符号前缀的概率之和
                alpha[t, r] = alpha[t-1, r] + alpha[t-1, r-1]
                # 如果可以跳跃连接，增加跳跃连接的前向概率
                if (r > 1 and skip_connect[r]):
                    alpha[t, r] += alpha[t-1, r-2]
                # 乘以当前时间步的对数概率
                alpha[t, r] *= logits[t, extended_symbols[r]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """计算后向概率。

        输入:
        -----
        logits: (np.array, 维度 = (input_len, len(symbols)))
                预测的（对数）概率

                在时间步t获取符号i的对数概率:
                p(t,s(i)) = logits[t,extended_symbols[i]]

        extSymbols: (np.array, 维度 = (2 * target_len + 1,))
                    扩展了空白符的标签序列

        skipConnect: (np.array, 维度 = (2 * target_len + 1,))
                    跳跃连接

        返回:
        ------
        beta: (np.array, 维度 = (input_len, 2 * target_len + 1))
                后向概率
    
        """

        S, T = len(extended_symbols), len(logits)  # S为扩展后的序列长度，T为输入序列长度
        betahat = np.zeros(shape=(T, S))  # 初始化后向概率的中间值矩阵
        beta = np.zeros(shape=(T, S))  # 初始化后向概率矩阵

        # 最后一个时间步的初始后向概率
        betahat[T-1, S-1] = logits[T-1, extended_symbols[S-1]]
        betahat[T-1, S-2] = logits[T-1, extended_symbols[S-2]]

        for t in range(T-2, -1, -1):
            # 计算时间步t的最后一个符号的后向概率
            betahat[t, S-1] = betahat[t+1, S-1] * logits[t, extended_symbols[S-1]]
            for r in range(S-2, -1, -1):
                # 当前符号的后向概率等于后一个符号和当前符号后缀的概率之和
                betahat[t, r] = betahat[t+1, r] + betahat[t+1, r+1]
                # 如果可以跳跃连接，增加跳跃连接的后向概率
                if (r <= S-3 and skip_connect[r+2]):
                    betahat[t, r] += betahat[t+1, r+2]
                # 乘以当前时间步的对数概率
                betahat[t, r] *= logits[t, extended_symbols[r]]

        # 归一化后向概率
        for t in range(T-1, -1, -1):
            for r in range(S-1, -1, -1):
                beta[t, r] = betahat[t, r] / logits[t, extended_symbols[r]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """计算后验概率。

        输入:
        -----
        alpha: (np.array, 维度 = (input_len, 2 * target_len + 1))
                前向概率

        beta: (np.array, 维度 = (input_len, 2 * target_len + 1))
                后向概率

        返回:
        ------
        gamma: (np.array, 维度 = (input_len, 2 * target_len + 1))
                后验概率
        """

        [T, S] = alpha.shape  # 获取时间步长度和扩展序列长度
        gamma = np.zeros(shape=(T, S))  # 初始化后验概率矩阵
        sumgamma = np.zeros((T,))  # 初始化后验概率的归一化因子

        for t in range(T):
            # 计算时间步t的后验概率
            for r in range(S):
                gamma[t, r] = alpha[t, r] * beta[t, r]
                sumgamma[t] += gamma[t, r]
            
            # 对时间步t的后验概率进行归一化
            for r in range(S):
                gamma[t, r] = gamma[t, r] / sumgamma[t]

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """
        初始化实例变量

        参数:
        -----------
        BLANK (int, 可选): 空白标签的索引。默认值为0。
        
        """
        super(CTCLoss, self).__init__()  # 调用父类的初始化函数

        self.BLANK = BLANK
        self.gammas = []  # 存储后验概率
        self.ctc = CTC()  # 初始化CTC对象

    def __call__(self, logits, target, input_lengths, target_lengths):
        # 调用forward函数计算CTC损失
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC 损失前向计算

        计算CTC损失，通过计算前向、后向和后验概率，然后计算目标和预测对数概率之间的平均损失。

        输入:
        -----
        logits [np.array, 维度=(seq_length, batch_size, len(symbols))]:
            RNN/GRU输出的对数概率（输出序列）

        target [np.array, 维度=(batch_size, padded_target_len)]:
            目标序列

        input_lengths [np.array, 维度=(batch_size,)]:
            输入序列的长度

        target_lengths [np.array, 维度=(batch_size,)]:
            目标序列的长度

        返回:
        -------
        loss [float]:
            后验概率和目标之间的平均散度
        """

        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        B, _ = target.shape  # 获取批次大小
        total_loss = np.zeros(B)  # 初始化总损失
        self.extended_symbols = []  # 初始化扩展符号列表

        for batch_itr in range(B):
            # -------------------------------------------->
            # 计算单个批次的CTC损失
            # 过程:
            #     将目标序列截断为目标长度
            #     将logits截断为输入长度
            #     用空白符扩展目标序列
            #     计算前向概率
            #     计算后向概率
            #     使用总概率函数计算后验概率
            #     计算每个批次的期望散度并存储在total_loss中
            #     对所有批次取平均并返回最终结果
            # <---------------------------------------------

            ctc = CTC()
            target_trunc = target[batch_itr, :target_lengths[batch_itr]]  # 截断目标序列到目标长度
            logits_trunc = logits[:input_lengths[batch_itr], batch_itr, :]  # 截断logits到输入长度
            extended_symbols, skip_connect = ctc.extend_target_with_blank(target_trunc)  # 扩展目标序列
            alpha = ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)  # 计算前向概率
            beta = ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)  # 计算后向概率
            gamma = ctc.get_posterior_probs(alpha, beta)  # 计算后验概率

            div = 0
            S, T = len(extended_symbols), len(logits_trunc)
            for t in range(T):
                for r in range(S):
                    # 计算后验概率和对数概率之间的散度
                    div -= gamma[t, r] * np.log(logits_trunc[t, extended_symbols[r]])
            
            total_loss += div  # 累加散度

        total_loss /= B  # 对所有批次取平均

        return total_loss

    def backward(self):
        """
        CTC损失反向计算

        计算相对于参数的梯度，并返回相对于输入（xt和ht）的导数。

        输入:
        -----
        logits [np.array, 维度=(seqlength, batch_size, len(Symbols))]:
            RNN/GRU输出的对数概率（输出序列）

        target [np.array, 维度=(batch_size, padded_target_len)]:
            目标序列

        input_lengths [np.array, 维度=(batch_size,)]:
            输入序列的长度

        target_lengths [np.array, 维度=(batch_size,)]:
            目标序列的长度

        返回:
        -------
        dY [np.array, 维度=(seq_length, batch_size, len(extended_symbols))]:
            散度相对于输入符号在每个时间步的导数
        """

        T, B, C = self.logits.shape  # 获取时间步、批次大小和符号数
        dY = np.full_like(self.logits, 0)  # 初始化导数矩阵

        for batch_itr in range(B):
            # -------------------------------------------->
            # 计算单个批次的CTC导数
            # 过程:
            #     将目标序列截断为目标长度
            #     将logits截断为输入长度
            #     用空白符扩展目标序列
            #     计算散度的导数并存储在dY中
            # <---------------------------------------------

            ctc = CTC()
            target_trunc = self.target[batch_itr, :self.target_lengths[batch_itr]]  # 截断目标序列到目标长度
            logits_trunc = self.logits[:self.input_lengths[batch_itr], batch_itr, :]  # 截断logits到输入长度
            extended_symbols, skip_connect = ctc.extend_target_with_blank(target_trunc)  # 扩展目标序列
            alpha = ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)  # 计算前向概率
            beta = ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)  # 计算后向概率
            gamma = ctc.get_posterior_probs(alpha, beta)  # 计算后验概率

            S, T = len(extended_symbols), len(logits_trunc)
            for t in range(T):
                for r in range(S):
                    # 计算散度相对于输入符号的导数
                    dY[t, batch_itr, extended_symbols[r]] -= gamma[t, r] / logits_trunc[t, extended_symbols[r]]

        return dY
