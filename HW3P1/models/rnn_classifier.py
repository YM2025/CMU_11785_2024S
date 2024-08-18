import numpy as np
import sys

# 添加自定义模块路径
sys.path.append("mytorch")
from rnn_cell import *  # 导入自定义的 RNNCell 类
from nn.linear import *  # 导入自定义的 Linear 类

class RNNPhonemeClassifier(object):
    """RNN 音素分类器类"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """
        初始化分类器

        参数:
        ---------
        input_size: int
            输入特征的维度
        hidden_size: int
            隐藏层单元的数量（隐藏状态的维度）
        output_size: int
            输出特征的维度
        num_layers: int, 默认值为2
            RNN 层的数量
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 初始化 RNN 层。第一层的输入维度是 input_size，后续层的输入维度是 hidden_size
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)  # 初始化输出层

        # 存储每个时间步的隐藏状态，形状为 [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """
        初始化权重

        参数:
        ---------
        rnn_weights: list
            包含多个 RNN 层权重的列表，每一层包括 [W_ih, W_hh, b_ih, b_hh]

        linear_weights: list
            包含线性层权重的列表 [W, b]
        """
        # 初始化每一层 RNN 的权重
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i]) # * 用于将一个列表或元组中的元素解包为函数的多个参数。
        # 初始化输出层的权重
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """
        RNN 前向传播，多层、多时间步。

        参数:
        ---------
        x: np.array
            输入数据，形状为 (batch_size, seq_len, input_size)

        h_0: np.array, 可选
            初始隐藏状态，形状为 (num_layers, batch_size, hidden_size)。如果未指定，则默认为全零

        返回:
        ---------
        logits: np.array
            输出的 logits，形状为 (batch_size, output_size)
        """
        # 获取批次大小和序列长度，初始化隐藏状态向量
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # 保存输入数据，并将初始隐藏状态添加到 hiddens 列表中
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        # 遍历序列中的每个时间步
        for i in range(seq_len):
            hidden_prev = hidden  # 保留前一个时间步的隐藏状态
            # 遍历每一层 RNN
            for l, rnn_cell in enumerate(self.rnn):
                if l == 0:
                    hidden[l,:,:] = rnn_cell(x[:, i, :], hidden_prev[l,:,:])  # 第1层使用输入数据
                else:
                    hidden[l,:,:] = rnn_cell(hidden[l-1,:,:], hidden_prev[l,:,:])  # 其余层使用上一层的隐藏状态
            self.hiddens.append(hidden.copy())  # 将当前时间步的隐藏状态添加到 hiddens 列表中

        # 最后一个时间步的隐藏状态通过线性层计算输出
        logits = self.output_layer(hidden[-1,:,:])
        
        return logits  # 返回 logits 作为模型输出



    def backward(self, delta):
        """
        RNN 反向传播，通过时间的反向传播（BPTT）。

        参数:
        ---------
        delta: np.array
            损失函数相对于最后一个时间步输出的梯度，形状为 (batch_size, hidden_size)

        返回:
        ---------
        dh_0: np.array
            损失函数相对于初始隐藏状态的梯度，形状为 (num_layers, batch_size, hidden_size)
        """
        # 初始化
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        
        # 从最后的输出层开始反向传播
        # dh[-1] 取出了 dh 的最后一层，形状为 (batch_size, self.hidden_size)
        dh[-1] = self.output_layer.backward(delta)  


        """
        * 注意:
        * 更详细的伪代码可能会在讲义幻灯片中出现，且在文档中有可视化的描述。
        * 小心由于实现决策导致的 off by 1 错误（例如索引偏移一位的错误）。

        伪代码:
        * 以时间顺序的逆序遍历（从 seq_len-1 到 0）
            * 以层次的逆序遍历（从 num_layers-1 到 0）
                * 根据当前层次从 hiddens 或 x 中获取 h_prev_l
                    （注意，hiddens 中包含一个额外的初始隐藏状态）
                * 使用 dh 和 hiddens 来获取反向传播方法的其他参数
                    （注意，hiddens 中包含一个额外的初始隐藏状态）
                * 使用 RNN 单元的反向传播结果更新 dh
                * 如果当前不是第一层，则需要将 dx 添加到第 l-1 层的梯度中。

        * 由于初始隐藏状态也被视为网络的参数，因此需要将 dh 除以 batch_size 进行归一化（即除以批次大小）。

        提示: 在某些地方可能需要使用 `+=` 操作。思考后再编写代码。
        """

        # 反向遍历时间步，从后向前
        for i in range(seq_len-1, -1, -1):
            # 反向遍历层次，从最后一层向前
            for l in range(self.num_layers-1, -1, -1):
                if l == 0:
                    h_prev_l = self.x[:, i, :]  # 如果是第1层，使用输入数据作为 h_prev_l
                else:
                    h_prev_l = self.hiddens[i+1][l-1,:,:]  # 其余层使用上一层的隐藏状态
                h_t = self.hiddens[i+1][l,:,:]  # 当前时间步的隐藏状态
                h_prev_t = self.hiddens[i][l,:,:]  # 前一个时间步的隐藏状态
                dx, dh_prev_t = self.rnn[l].backward(dh[l,:,:], h_t, h_prev_l, h_prev_t)  # 执行 RNN 单元的反向传播
                dh[l,:,:] = dh_prev_t  # 更新当前层的梯度
                if l > 0:
                    dh[l-1,:,:] += dx  # 如果不是第1层，将 dx 加到上一层的梯度中

        return dh / batch_size  # 返回相对于初始隐藏状态的平均梯度
