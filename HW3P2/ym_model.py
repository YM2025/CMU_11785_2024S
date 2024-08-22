
import torch  
import random  
import numpy as np  
import torch.nn as nn  # 导入torch.nn模块，用于构建神经网络层
import torch.nn.functional as F  # 导入torch.nn.functional模块，包含常用的神经网络函数

# from torchsummaryX import summary  # 导入torchsummaryX库的summary函数
from torchinfo import summary  # 导入torchinfo库的summary函数，用于显示模型的结构信息
from torch.utils.data import Dataset, DataLoader  # 导入Dataset和DataLoader，用于处理和加载数据
from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_padded_sequence, pad_packed_sequence  # 导入RNN序列处理相关的工具

import torchaudio.transforms as tat  # 导入torchaudio.transforms模块，包含音频处理的工具

from sklearn.metrics import accuracy_score  # 导入accuracy_score函数，用于计算准确率
import gc  # 导入gc模块，用于进行垃圾回收

# import zipfile  # 导入zipfile模块，用于处理ZIP文件
import pandas as pd  
from tqdm import tqdm  # 导入tqdm库，用于显示循环进度条
import os  
import datetime  # 导入datetime模块，用于处理日期和时间

# 导入解码和距离计算的库
# `ctcdecode` 是一个已废弃的包，它不能在现代编译器上进行编译
# import ctcdecode  # 导入ctcdecode模块，用于CTC解码
import Levenshtein  # 导入Levenshtein模块，用于计算Levenshtein距离
# from ctcdecode import CTCBeamDecoder  # 从ctcdecode模块导入CTCBeamDecoder类
from torchaudio.models.decoder import ctc_decoder  # 从torchaudio.models.decoder模块导入ctc_decoder，用于CTC解码

# from torchnlp.nn import LockedDropout  # 导入 LockedDropout 模块



from ym_config import *
from ym_dataset import *



class PermuteBlock(torch.nn.Module):  
    def forward(self, x):  # 定义前向传播函数forward，该函数在每次前向传播时被自动调用
        return x.transpose(1, 2)  # 将输入张量x的第1维和第2维进行转置并返回


class LockedDropout(nn.Module):
    def __init__(self, drop_prob):
        super(LockedDropout, self).__init__()
        self.prob = drop_prob
    def forward(self, x):
        if not self.training or not self.prob: # turn it off during inference
            return x
        x, x_lens = pad_packed_sequence(x, batch_first = True)
        m = x.new_empty(x.size(0), 1, x.size(2),requires_grad=False).bernoulli_(1 - self.prob)
        mask = m / (1 - self.prob)
        mask = mask.expand_as(x)
        out = x * mask
        out = pack_padded_sequence(out,x_lens, batch_first = True, enforce_sorted= False)
        return out




"""
Encoder 类实现了一个音频编码器，用于将输入的音频特征序列转换为更高层次的表示。
使用卷积层和批归一化层对输入进行特征提取和归一化。
使用金字塔双向LSTM层（pBLSTM）处理序列数据，结合 LockedDropout 防止过拟合。
"""
class Encoder(nn.Module):  # 定义 Encoder 类，继承自 nn.Module
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        expand_dims = [128, 256]  # 定义卷积层的输出维度
                
        self.embed = nn.Sequential(
            PermuteBlock(),  # 调整输入的维度顺序。[batchsize, 最大帧数, 27] → [batchsize, 27, 最大帧数]
            nn.Conv1d(in_channels=input_size, out_channels=expand_dims[0], kernel_size=3, stride=1, padding=1),  # 1D卷积操作在时间维度上进行滑动窗口操作，提取跨时间步的局部特征。 [batchsize, 128, 最大帧数]
            nn.BatchNorm1d(num_features=expand_dims[0]),  # BN 对每个特征通道（128）独立地进行标准化操作
            nn.GELU(),  
            nn.Conv1d(in_channels=expand_dims[0], out_channels=expand_dims[1], kernel_size=3, stride=1, padding=1),  # [batchsize, 256, 最大帧数]
            nn.BatchNorm1d(num_features=expand_dims[1]),  # BN 对每个特征通道（256）独立地进行标准化操作
            PermuteBlock()  # 调整维度顺序
        )

        self.pBLSTMs = nn.Sequential(
            pBLSTM(input_size=expand_dims[1], hidden_size=hidden_size),  # 第一个金字塔双向LSTM层
            LockedDropout(0.4),  # 使用 LockedDropout，dropout 率为 0.4
            pBLSTM(input_size=2*hidden_size, hidden_size=hidden_size),  # 第二个金字塔双向LSTM层
            LockedDropout(0.3)  # 使用 LockedDropout，dropout 率为 0.3
        )
         
    def forward(self, x, lens):
        x = self.embed(x)  # 通过嵌入层。输出维度 [batchsize, 最大帧数， 256]
        lens = lens.clamp(max=x.shape[1]).cpu()  # lens维度是（batchsize， ），每个元素的值代表一个样本的原始帧数。这行代码将 lens 中的所有值都限制在 x.shape[1] 的最大值，避免后续索操作引越界。
        
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)  # 打包输入。 x 不再是一个传统的张量，而是一个 PackedSequence 对象，专门为 RNN 等序列模型设计，以更有效地处理变长序列。
        x = self.pBLSTMs(x)  # 通过金字塔双向LSTM层
        outputs, lens = pad_packed_sequence(x, batch_first=True)  # 解包输出

        return outputs, lens  # 返回输出和长度







class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            PermuteBlock(), torch.nn.BatchNorm1d(embed_size), PermuteBlock(),
            #TODO define your MLP arch. Refer HW1P2
            #Use Permute Block before and after BatchNorm1d() to match the size
            nn.Linear(embed_size, 2048),
            nn.GELU(),
            PermuteBlock(), torch.nn.BatchNorm1d(2048), PermuteBlock(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.GELU(),
            PermuteBlock(), torch.nn.BatchNorm1d(1024), PermuteBlock(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_size)
        )
        
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        #TODO call your MLP
        #TODO Think what should be the final output of the decoder for the classification 
        out = self.mlp(encoder_out)
        out = self.softmax(out)

        return out 





class pBLSTM(torch.nn.Module):
    '''
    Pyramidal BiLSTM (金字塔双向LSTM)
    请阅读相关文献并理解其概念，然后在此编写实现。

    每个步骤的实现：
    1. 如果输入是打包的 (PackedSequence)，需要先对其进行填充 (Unpack)
    2. 通过连接特征维度来减少输入的时间长度
        (提示: 写下张量的形状来理解)
        (i) 如何处理奇数/偶数长度的输入？
        (ii) 在截断输入后，如何处理输入长度数组 (x_lens)？
    3. 再次打包输入
    4. 将处理后的输入传入LSTM层

    为了使我们的实现具有模块化，每次只传递一层。
    '''
    
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        # 定义LSTM层，输入维度是原始输入维度的2倍，隐藏层大小为hidden_size，双向LSTM
        self.blstm = nn.LSTM(input_size=2*input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True, dropout=0.2, batch_first=True) 

    def forward(self, x_packed):  # x_packed 是打包后的序列 (PackedSequence)
        
        # 将打包的序列解包
        x, lengths = pad_packed_sequence(x_packed, batch_first=True)
        
        # 两两拼接时间步（帧），特征数翻倍。
        x, x_lens = self.trunc_reshape(x, lengths)
        
        # 将调整后的输入重新打包
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        
        # 将处理后的输入传入LSTM层
        x, h = self.blstm(x)
        
        return x

    def trunc_reshape(self, x, x_lens): 
        # 如果输入的时间维度是奇数，截去最后一个时间步
        if x.shape[1] % 2 != 0:
            x = x[:, :-1, :]

        # 重新调整输入的形状，将时间维度减半，特征维度加倍
        x = x.reshape(x.shape[0], x.shape[1] // 2, x.shape[2] * 2)
        # 更新输入的时间长度
        x_lens = x_lens // 2
        return x, x_lens



  
    
    
class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size=192, output_size=len(PHONEMES)):
        super().__init__()

        ## 可选的数据增强部分
        # self.augmentations  = torch.nn.Sequential(
        #     # 在这里添加时间掩码和频率掩码
        #     PermuteBlock(), 
        #     torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
        #     torchaudio.transforms.TimeMasking(time_mask_param=100),
        #     PermuteBlock(),
        # ) 
        # 数据增强在 collate_fn 中进行，所以这里注释掉
        self.encoder = Encoder(input_size, embed_size)  # 初始化编码器
        self.decoder = Decoder(embed_size*2, output_size)  # 初始化解码器
    
    def forward(self, x, lengths_x):
        # 在训练模式下使用数据增强
        # if self.training:
        #     x = self.augmentations(x)
        
        # x = [512, 1723, 27]（512是batchsize, 1723是512个样本中最大的帧数）   lengths_x = [512]（每个值是每个样本的总帧数）
        encoder_out, encoder_lens = self.encoder(x, lengths_x)  # 编码器前向传播
        decoder_out = self.decoder(encoder_out)  # 解码器前向传播

        return decoder_out, encoder_lens  # 返回解码器输出和编码器长度






