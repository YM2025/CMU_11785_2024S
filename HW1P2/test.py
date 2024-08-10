import torch
import numpy as np
import random
from torchsummaryX import summary
import sklearn
import gc #垃圾回收模块（Garbage Collector）的语句。垃圾回收是Python内存管理的一部分，用于自动回收不再使用的内存空间。
import zipfile
import pandas as pd
from tqdm.auto import tqdm
import os
import datetime 
import wandb 

from torch.cuda.amp import GradScaler, autocast # 使用自动混合精度
scaler = GradScaler() # 初始化梯度缩放器，用于自动混合精度训练

# 导入自定义的数据集和模型
from dataset import AudioDataset, PHONEMES
from model import MLP_Net, config

from help import should_switch_to_sgd, compute_kl_loss


# 定义所有库的随机种子。
SEED = 11785
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AudioTestDataset(torch.utils.data.Dataset): 
    def __init__(self, root, phonemes = PHONEMES, context=0, partition= "test-clean", use_cmn=False): 
        self.context    = context
        self.phonemes   = phonemes[:-2]
        self.use_cmn = use_cmn
        self.PHONEMES_MAP = {}
        for i, p in enumerate(self.phonemes): #将音素list转成字典。
            self.PHONEMES_MAP[p] = i
        
        self.mfcc_dir       = root+partition+'/mfcc/' 
        mfcc_names          =  sorted(os.listdir(self.mfcc_dir))

        self.mfccs = []
        for i in tqdm(range(len(mfcc_names))): 
            mfcc        = np.load(self.mfcc_dir + mfcc_names[i])
            if self.use_cmn:
                mean = np.mean(mfcc, axis=0) # 计算每个特征的均值
                std = np.std(mfcc, axis=0)
                mfcc = (mfcc - mean) / std 
            self.mfccs.append(mfcc)

        self.mfccs  = np.concatenate(self.mfccs,  axis = 0)
        self.length = len(self.mfccs)
        self.mfccs = np.pad(self.mfccs, ((self.context, self.context), (0,0)), 'constant', constant_values = (0,0))

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        # 实际索引需要加上self.context以考虑预填充的零
        actual_idx = ind + self.context

        # 计算实际的上下文窗口的边界
        lower = actual_idx - self.context  # 这里可以安全地使用，因为前面有self.context个零
        upper = actual_idx + self.context + 1

        # 从mfccs中切片得到帧和其上下文
        frames = self.mfccs[lower:upper]
        frames = frames.flatten() # TODO: Flatten to get 1d data
        frames      = torch.FloatTensor(frames) 
        return frames


def test(model, test_loader):
    model.eval() 
    test_predictions = []

    with torch.no_grad():
        for i, mfccs in enumerate(tqdm(test_loader)):

            mfccs   = mfccs.to(device)             
            logits  = model(mfccs)

            predicted_phonemes = torch.argmax(logits, dim= 1)
            predicted_phonemes_array = predicted_phonemes.cpu().detach().numpy()

            test_predictions = test_predictions + predicted_phonemes_array.tolist()

    return test_predictions






if __name__ == '__main__':
    # 加载数据集
    data_path = 'E:/cv_data/cmu11785_data/'
    test_data = AudioTestDataset(data_path, 
                                 partition="test-clean", 
                                 phonemes = PHONEMES, 
                                 context=config['context'], 
                                 use_cmn=config['use_cmn'],
                                 )

    test_loader = torch.utils.data.DataLoader(
        dataset     = test_data,
        num_workers = 4,
        batch_size  = config['batch_size'],
        pin_memory  = True,
        shuffle     = False
    )


    INPUT_SIZE  = (2*config['context'] + 1) * 28  # 1表示MFCC一帧（25ms），28表示每帧的MFCC特征数，2*context+1表示上下文窗口大小
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化模型
    model       = MLP_Net(INPUT_SIZE, len(test_data.phonemes), 
                          config['hidden_sizes'], 
                          config['dropout_rate'], 
                          use_LeakyReLU = config['use_LeakyReLU'], 
                          use_batchnorm = config['use_batchnorm'], 
                          use_dropout = config['use_dropout'], 
                          custom_weight_init = config['custom_weight_init']).to(device)
    
    # 加载已经训练好的模型权重
    model_path = './model.pt'  
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # 正确加载模型权重部分
    
    # 前向传播
    predictions = test(model, test_loader)

    # 保存结果文件
    with open("./submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(predictions)):
            f.write("{},{}\n".format(i, predictions[i]))