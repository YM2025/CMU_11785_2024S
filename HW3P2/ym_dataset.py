from ym_config import *
from ym_model import * 


import torch  
import random  
import numpy as np  
import torch.nn as nn  # 导入torch.nn模块，用于构建神经网络层
import torch.nn.functional as F  # 导入torch.nn.functional模块，包含常用的神经网络函数

# from torchsummaryX import summary  # 导入torchsummaryX库的summary函数
from torchinfo import summary  # 导入torchinfo库的summary函数，用于显示模型的结构信息
from torch.utils.data import Dataset, DataLoader  # 导入Dataset和DataLoader，用于处理和加载数据
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence  # 导入RNN序列处理相关的工具

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

from torchnlp.nn import LockedDropout  # 导入 LockedDropout 模块



class AudioDataset(torch.utils.data.Dataset):  # 定义一个继承自torch.utils.data.Dataset的自定义数据集类

    def __init__(self, root=DATA_ROOT, partition="train-clean-100", use_cmn=False, audio_transformation=None):
        '''
        初始化数据集。

        输入参数:
        root: 数据存储的根目录
        partition: 数据集的划分部分（例如训练集）
        use_cmn: 是否使用倒谱均值归一化 (Cepstral Mean Normalization)
        audio_transformation: 音频数据的变换操作，如果未指定则不进行任何变换
        '''

        # 加载目录及其所有文件
        self.phonemes = PHONEMES  # 定义音素列表
        self.mfccs, self.transcripts = self._init_data(f"{root}/{partition}", use_cmn=use_cmn)  # 初始化MFCC和转录数据

        print(self.length, len(self.mfccs), len(self.transcripts))  # 输出数据集长度信息

        if audio_transformation is not None:
            self.transformation = audio_transformation  # 如果提供了音频变换操作，则使用该操作
        else:
            self.transformation = nn.Sequential()  # 否则不进行任何变换

        # 此处可以创建音素到索引的映射，以便将音素表示为数字，方便存储和处理

    def _init_data(self, root: str, use_cmn=False):
        self.mfcc_dir = f"{root}/mfcc"  # MFCC数据的目录
        self.transcript_dir = f"{root}/transcript"  # 转录数据的目录
        mfcc_names = os.listdir(self.mfcc_dir)  # 获取MFCC文件名列表
        transcript_names = os.listdir(self.transcript_dir)  # 获取转录文件名列表
        
        self.length = len(mfcc_names)  # 获取数据集文件的个数

        self.mfccs, self.transcripts = [], []
        for i in tqdm(range(len(mfcc_names))):
            mfcc = np.load(f"{self.mfcc_dir}/{mfcc_names[i]}")  # 加载一个MFCC文件
            if use_cmn:
                mfcc = mfcc - np.mean(mfcc, axis=0)  # 如果启用，执行倒谱均值归一化
            transcript = np.load(f"{self.transcript_dir}/{transcript_names[i]}")  # 加载相应的转录文件
            assert transcript[0] == '[SOS]' and transcript[-1] == '[EOS]'  # 确保转录的开始和结束标记
            transcript = transcript[1:-1]  # 移除[SOS]和[EOS]标记
            transcript = np.vectorize(self.phonemes.index)(transcript)  # 将转录中的音素转换为索引
            self.mfccs.append(mfcc)  # 将处理后的MFCC添加到列表中
            self.transcripts.append(transcript)  # 将处理后的转录添加到列表中
            
        return self.mfccs, self.transcripts  # 返回处理后的MFCC和转录列表

    def __len__(self):
        return self.length  # 返回数据集的总长度

    def __getitem__(self, ind):
        '''
        返回指定索引的数据（MFCC和对应的标签）。

        输入:
        ind: 数据的索引

        输出:
        返回包含MFCC特征和对应标签的元组。
        '''
        mfcc = self.mfccs[ind]  # 获取指定索引的MFCC数据
        transcript = self.transcripts[ind]  # 获取对应的转录标签
        return torch.FloatTensor(mfcc), torch.tensor(transcript)  # 转换为张量并返回

    def collate_fn(self, batch):
        '''
        将批量数据处理为适合模型输入的形式。

        输入:
        batch: 一批数据

        输出:
        返回填充后的MFCC特征、填充后的标签、特征的实际长度、标签的实际长度。
        '''
        batch_mfcc, batch_transcript = [], []  # 用于存储批量的MFCC和转录数据
        lengths_mfcc, lengths_transcript = [], []  # 用于存储每个数据的实际长度
        for (m, t) in batch:
            batch_mfcc.append(m)  # 将MFCC数据添加到批量列表中
            lengths_mfcc.append(len(m))  # 存储本MFCC样本的实际帧个数
            batch_transcript.append(t)  # 将转录数据添加到批量列表中
            lengths_transcript.append(len(t))  # 记录转录数据的实际长度
        
        # pad_sequence 用于对不定长的序列进行填充（padding），以便能够将它们组合成一个批次（batch）进行并行计算。较短的序列会在结尾填充 0，以匹配最长的序列。
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)  # 对MFCC数据进行填充
        batch_mfcc_pad = self.transformation(batch_mfcc_pad)  # 对填充后的MFCC数据进行变换操作（如果有）
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)  # 对转录数据进行填充

        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)  # 返回处理后的数据



# 你可以将此作为参数传递给上面的数据集类
# 这将有助于模块化你的实现
transforms = nn.Sequential(
    # 备注：进入的维度形如[512, 1705, 27] ，（batch，最大帧数，MFCC特征数）
    tat.FrequencyMasking(freq_mask_param=5),  # 频率掩码，用于随机遮挡某些频率段
    tat.TimeMasking(time_mask_param=35),  # 时间掩码，用于随机遮挡某些时间段
)  




class AudioDatasetTest(AudioDataset):  # 定义一个继承自AudioDataset的测试数据集类
    def __init__(self, root=DATA_ROOT, partition="test-clean", use_cmn=True, audio_transformation=None):
        # 调用父类的初始化方法，设置数据集的根目录、划分部分（默认为“test-clean”）等
        super().__init__(root, partition, use_cmn, audio_transformation=None)
      
    # 加载MFCC数据，并为测试集创建空白的转录标签。  
    def _init_data(self, root: str, use_cmn):
        # 初始化数据，加载MFCC文件
        self.mfcc_dir = f"{root}/mfcc"  # 设置MFCC文件的目录

        mfcc_names = os.listdir(self.mfcc_dir)  # 获取MFCC文件名列表
        
        self.length = len(mfcc_names)  # 设置数据集的长度

        self.mfccs, self.transcripts = [], []  # 初始化MFCC和转录列表

        for i in tqdm(range(len(mfcc_names))):
            mfcc = np.load(f"{self.mfcc_dir}/{mfcc_names[i]}")  # 加载单个MFCC文件
            transcript = np.array([0 for _ in range(len(mfcc))])  # 创建与MFCC长度相同的空白转录

            assert len(mfcc) == len(transcript)  # 确保MFCC和转录长度一致

            self.mfccs.append(mfcc)  # 将MFCC添加到列表中
            self.transcripts.append(transcript)  # 将转录添加到列表中

        return np.concatenate(self.mfccs, axis=0), np.concatenate(self.transcripts, axis=0)  # 返回合并的MFCC和转录数据
      
    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]  # 获取指定索引的MFCC数据
        return torch.FloatTensor(mfcc)  # 将MFCC转换为张量并返回
      
    def collate_fn(self, batch):
        batch_mfcc = []  # 初始化批量MFCC列表
        lengths_mfcc = []  # 初始化MFCC长度列表
        
        for mfcc in batch:
            batch_mfcc.append(mfcc)  # 将MFCC添加到批量列表中
            lengths_mfcc.append(len(mfcc))  # 记录MFCC的实际长度
      
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)  # 对MFCC进行填充
        
        # 这里可以对批量数据应用一些变换操作，如时间和频率掩码
        
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)  # 返回填充后的MFCC数据和实际长度
