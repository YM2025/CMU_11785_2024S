import os
import numpy as np
import torch
import random
from tqdm import tqdm

# 音素列表，包含所有音素和特殊标记，共42个
PHONEMES = [
            '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]'
        ]

class AudioDataset(torch.utils.data.Dataset):
    """
    数据集类用于加载训练和验证数据。

    Args:
        root (str): 数据集的根目录路径。
        phonemes (list): 音素列表，默认为PHONEMES。
        context (int): 上下文窗口大小，决定了从当前时间步前后各取多少帧。
        partition (str): 数据分区，默认为"train-clean-100"。
        subset_ratio：数据集使用比率。最大1.0
        use_mask：是否对数据进行 mask。
        use_cmn：是否对数据进行 cepstral mean normalization 归一化。

    Attributes:
        context (int): 上下文窗口大小。
        phonemes (list): 音素列表。
        mfcc_dir (str): MFCC文件的目录路径。
        transcript_dir (str): 语音转录文件的目录路径。
        mfccs (ndarray): 存储所有MFCC特征的数组。
        transcripts (ndarray): 存储所有转录的数组。
        length (int): 数据集的总时长

    Methods:
        __len__(): 返回数据集的长度。
        __getitem__(ind): 根据索引获取数据集中的样本。

    """

    def __init__(self, root, phonemes=PHONEMES, context=0, partition="train-clean-100", subset_ratio=1.0, use_mask=0, use_cmn=False):
        self.use_mask = use_mask
        self.use_cmn = use_cmn
        # 初始化上下文窗口大小、音素列表和数据分区
        self.context = context
        self.phonemes = phonemes[:-2] #去除最后的两个非音素类标记。
        self.PHONEMES_MAP = {}
        for i, p in enumerate(self.phonemes): #将音素list转成字典。
            self.PHONEMES_MAP[p] = i

        # 设置MFCC和语音转录文件的目录路径
        self.mfcc_dir = root + partition + '/mfcc/'
        self.transcript_dir = root + partition + '/transcript/'

        # 获取 MFCC 和 转录文件 的文件名列表，并确保两者数量相同
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))
        assert len(mfcc_names) == len(transcript_names)

        # 应用subset_ratio来决定使用多少数据
        mfcc_length = int(len(mfcc_names) * subset_ratio)
        mfcc_names = mfcc_names[:mfcc_length]
        transcript_names = transcript_names[:mfcc_length]

        # 计算数据集的总帧数
        T = 0
        for i in range(mfcc_length):
            mfcc = np.load(self.mfcc_dir + mfcc_names[i]) # 单个mfcc文件的维度：[帧数, 特征数]
            T += mfcc.shape[0]
        self.length = T # 数据集的总帧数（每帧25ms信息）
        
        # 将所有 MFCC 和 标签文件 全部加载到 self.mfccs 和 self.transcripts 中
        self._init_data(mfcc_names, transcript_names, mfcc.shape[1])


    def __len__(self):
        return self.length # 返回数据集的总帧数。
    

    #__getitem__方法是针对单个样本的操作。
    # 当你设置了batch_size，DataLoader（num_works启动的子进程数）会自动地将多个由__getitem__返回的样本组合成一个批次。
    def __getitem__(self, index):
        """
        根据索引获取数据集中的样本。

        参数:
            index (int): 样本的索引。

        返回:
            tuple: 包含 MFCC帧 和 对应的音素标签 的元组。

        """
        # 实际索引需要加上self.context以考虑预填充的零
        actual_idx = index + self.context

        # 计算实际的上下文窗口的边界
        lower = actual_idx - self.context  # 这里可以安全地使用，因为前面有self.context个零
        upper = actual_idx + self.context + 1

        # 从mfccs中切片得到帧和其上下文
        frames = self.mfccs[lower:upper]

        # 验证是否取得正确的窗口大小，通常为 2*self.context + 1
        assert len(frames) == 2 * self.context + 1, f"Expected window size of {2 * self.context + 1}, but got {len(frames)}"
            
        # 进行mask
        if self.use_mask:
            frames = self._apply_masking(frames)    
            
        # 展平frames以供MLP模型使用
        frames = frames.flatten()
        
        frames = torch.FloatTensor(frames)
        phonemes = torch.tensor(self.transcripts[index])  # 注意，transcripts应没有预填充

        return frames, phonemes
        
        
        
    
    def _init_data(self, mfcc_names, transcript_names, mfcc_Feature_num):
            
        self.mfccs = np.zeros((self.length + 2 * self.context, mfcc_Feature_num), dtype=np.float32) # ! 数组的内存空间一次性分配好，适合于处理大量数据时提高性能。
        self.transcripts = np.zeros((self.length,), dtype=np.uint8)
        
        # 开始将数据集填充到上述两个变量中
        mfccs_index, transcripts_index = 0, 0 
        for i in tqdm(range(len(mfcc_names))): # ! tqdm是第三方库，用于为长时间运行的循环提供可视化的进度指示。
            
            mfcc = np.load(self.mfcc_dir + mfcc_names[i]) # mfcc维度：[帧数，特征数]
            
            if self.use_cmn:
                mean = np.mean(mfcc, axis=0) # 计算每个特征的均值
                std = np.std(mfcc, axis=0)
                mfcc = (mfcc - mean) / std 
            
            transcript = np.load(self.transcript_dir + transcript_names[i])
            assert transcript[0] == '[SOS]' and transcript[-1] == '[EOS]'
            transcript = transcript[1:-1] #去除数据头尾的'[SOS]', '[EOS]'标记。 
            #将音素标记转换为数字序号
            # transcript原本是一个字符串列表，每个字符串是一个音素标记
            # 转换后的transcript是一个整数列表，每个整数代表音素的索引
            transcript = [self.PHONEMES_MAP[p] for p in transcript] # 现在的 transcript 维度：[帧数, ]，每个元素值代表音素的索引号。

            T_i = mfcc.shape[0] # 获取该样本的帧个数
            self.mfccs[mfccs_index : mfccs_index + T_i] = mfc
            self.transcripts[transcripts_index : transcripts_index + T_i] = transcript
            mfccs_index += T_i
            transcripts_index += T_i
        
        # 在首尾填充 self.context 个 0。
        self.mfccs = np.concatenate([np.zeros((self.context, mfcc_Feature_num)), self.mfccs, np.zeros((self.context, mfcc_Feature_num))], axis=0)
        
        
    # 
    def _apply_masking(self, frames): # frames维度: [帧数，特征数]
        mask_percentage = 0.05 # 10%几率频率或时间轴上mask
        mask_dropout = 0.05 # 所有特征，独立5% mask
        
        if self.use_mask == 1: # Dropout式特征掩蔽（所有帧上的所有特征，完全独立的5%几率置为0）
            mask = np.random.binomial(1, 1-mask_dropout, frames.shape)  # 生成dropout掩码
            return frames * mask 
        elif self.use_mask == 2: # 时间掩码
            seq_len, feature_dim = frames.shape
            mask_len = int(seq_len * mask_percentage)
            mask_indices = np.random.choice(seq_len, mask_len, replace=False)
            frames[mask_indices, :] = 0
        elif self.use_mask == 3: # 频率掩码
            seq_len, feature_dim = frames.shape
            mask_len = int(feature_dim * mask_percentage)
            mask_indices = np.random.choice(feature_dim, mask_len, replace=False)
            frames[:, mask_indices] = 0
        elif self.use_mask == 4: # 时间 + 频率掩码
            seq_len, feature_dim = frames.shape

            # 随机时间掩码
            time_mask_len = int(seq_len * mask_percentage)
            time_mask_indices = np.random.choice(seq_len, time_mask_len, replace=False)
            frames[time_mask_indices, :] = 0

            # 随机频率掩码
            freq_mask_len = int(feature_dim * mask_percentage)
            freq_mask_indices = np.random.choice(feature_dim, freq_mask_len, replace=False)
            frames[:, freq_mask_indices] = 0

        return frames


