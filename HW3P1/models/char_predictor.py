import numpy as np
import sys

# 添加自定义模块路径
sys.path.append("mytorch")
from gru_cell import *  # 导入自定义的 GRUCell 类
from nn.linear import *  # 导入自定义的 Linear 类

class CharacterPredictor(object):
    """CharacterPredictor 类。

    这个类实现了一个神经网络，可以处理输入序列的一个时间步。
    你只需要实现这个类的 forward 方法。
    这个类主要用于测试你实现的 GRU Cell 在作为 GRU 使用时是否正确。
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        """网络由一个 GRU Cell 和一个线性层组成。"""
        self.gru = GRUCell(input_dim, hidden_dim)  # 初始化 GRU Cell，输入维度为 input_dim，隐藏层维度为 hidden_dim
        self.projection = Linear(hidden_dim, num_classes)  # 初始化线性层，输入维度为 hidden_dim，输出维度为 num_classes
        self.num_classes = num_classes  # 保存类别数
        self.hidden_dim = hidden_dim  # 保存隐藏层维度
        self.projection.W = np.random.rand(num_classes, hidden_dim)  # 随机初始化线性层的权重

    def init_rnn_weights(
        self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
    ):
        """初始化 GRU 的权重，不需要修改这个函数。"""
        self.gru.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
        )

    def __call__(self, x, h):
        # 使对象实例可以像函数一样调用，实际调用的是 forward 方法
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor 的前向传播。

        处理输入序列的一个时间步。

        参数
        -----
        x: np.array, 形状为 (feature_dim)
            当前时间步的输入。

        h: np.array, 形状为 (hidden_dim)
            前一个时间步的隐藏状态。
        
        返回
        -------
        logits: np.array, 形状为 (num_classes)
            当前时间步的 logits 输出。

        hnext: np.array, 形状为 (hidden_dim)
            当前时间步的隐藏状态。

        """
        hnext = self.gru(x, h)  # 通过 GRU Cell 计算当前时间步的隐藏状态
        # self.projection 期望输入的形状为 (batch_size, input_dimension)，因此需要将 hnext 重塑为 (1, -1)
        logits = self.projection(hnext.reshape(1,-1))  # 通过线性层计算 logits
        logits = logits.reshape(-1,)  # 将 logits 重塑为 (num_classes)
        return logits, hnext  # 返回 logits 和当前时间步的隐藏状态 hnext


def inference(net, inputs):
    """CharacterPredictor 的推理函数。

    使用上面定义的类的一个实例，对输入序列进行推理，生成每个时间步的 logits 输出。

    参数
    -----
    net:
        CharacterPredictor 类的一个实例。

    inputs: np.array, 形状为 (seq_len, feature_dim)
            输入序列，每个时间步的输入维度为 feature_dim。

    返回
    -------
    logits: np.array, 形状为 (seq_len, num_classes)
            每个时间步的 logits 输出。

    """
    seq_len, feature_dim = inputs.shape  # 获取序列长度和输入维度
    logits = np.zeros((seq_len, net.num_classes))  # 初始化 logits 输出矩阵
    h = np.zeros(net.hidden_dim)  # 初始化隐藏状态为全零
    for i in range(seq_len):  # 遍历序列中的每个时间步
        logits[i, :], h = net(inputs[i, :], h)  # 对每个时间步进行推理，并更新隐藏状态
    
    return logits  # 返回所有时间步的 logits 输出
