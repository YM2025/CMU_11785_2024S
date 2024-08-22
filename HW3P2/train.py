import torch  
import random  
import numpy as np  
import torch.nn as nn  # 导入torch.nn模块，用于构建神经网络层
import torch.nn.functional as F  # 导入torch.nn.functional模块，包含常用的神经网络函数

# from torchsummaryX import summary  # 导入torchsummaryX库的summary函数
from torchinfo import summary  # 导入torchinfo库的summary函数，用于显示模型的结构信息
from torch.utils.data import Dataset, DataLoader  # 导入Dataset和DataLoader，用于处理和加载数据
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence  # 导入RNN序列处理相关的工具
import torchaudio
import torchaudio.transforms as tat  # 导入torchaudio.transforms模块，包含音频处理的工具

# 导入torchaudio的CTC解码器
from pyctcdecode import build_ctcdecoder  # 导入pyctcdecode库中的CTC解码器构建函数
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
from torchaudio.models.decoder import cuda_ctc_decoder

import wandb
import torchsummaryX


import warnings  # 导入warnings模块，用于处理警告信息
warnings.filterwarnings('ignore')  # 忽略所有警告信息

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print("Device: ", device) 


from ym_config import *
from ym_dataset import *
from ym_model import * 
from ym_helps import *




def train_model(model, train_loader, criterion, optimizer):

    model.train()  # 将模型设置为训练模式
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')  # 初始化进度条

    total_loss = 0  # 初始化总损失

    for i, data in enumerate(train_loader):  # 遍历训练数据集
        optimizer.zero_grad()  # 梯度清零

        x, y, lx, ly = data  # batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)
        x, y = x.to(device), y.to(device)  

        #with torch.cuda.amp.autocast():  # 使用自动混合精度加速训练
        with torch.amp.autocast("cuda"):
            h, lh = model(x, lx)  # 前向传播
            h = torch.permute(h, (1, 0, 2))  # 调整输出形状以适应CTC损失
            loss = criterion(h, y, lh, ly)  # 计算损失

        total_loss += loss.item()  # 累加损失

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),  # 更新平均损失
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr']))  # 更新学习率
        )

        batch_bar.update()  # 更新进度条

        # FP16 训练相关步骤
        scaler.scale(loss).backward()  # 反向传播
        scaler.step(optimizer)  # 更新模型参数
        scaler.update()  # 更新缩放器

        del x, y, lx, ly, h, lh, loss  # 释放内存
        torch.cuda.empty_cache()  # 清理GPU缓存

    batch_bar.close()  # 关闭进度条

    return total_loss / len(train_loader)  # 返回平均损失



def validate_model(model, val_loader, decoder, phoneme_map=LABELS):

    model.eval()  # 将模型设置为评估模式
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')  # 初始化进度条

    total_loss = 0  # 初始化总损失
    vdist = 0  # 初始化总的Levenshtein距离

    for i, data in enumerate(val_loader):  # 遍历验证数据集

        x, y, lx, ly = data  # 获取输入数据和标签
        x, y = x.to(device), y.to(device)  # 将数据移动到设备（GPU或CPU）

        with torch.inference_mode():  # 禁用梯度计算
            h, lh = model(x, lx)  # 前向传播
            h = torch.permute(h, (1, 0, 2))  # 调整输出形状以适应CTC损失
            loss = criterion(h, y, lh, ly)  # 计算损失

        total_loss += float(loss)  # 累加损失
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)  # 计算并累加Levenshtein距离

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),  # 更新平均损失
            dist="{:.04f}".format(float(vdist / (i + 1)))  # 更新平均Levenshtein距离
        )

        batch_bar.update()  # 更新进度条

        del x, y, lx, ly, h, lh, loss  # 释放内存
        torch.cuda.empty_cache()  # 清理GPU缓存

    batch_bar.close()  # 关闭进度条
    total_loss = total_loss / len(val_loader)  # 计算平均损失
    val_dist = vdist / len(val_loader)  # 计算平均Levenshtein距离
    return total_loss, val_dist  # 返回验证损失和Levenshtein距离







if __name__ == '__main__':
    
    # 1）wandb配置
    wandb.login(key="113991ce0fd655043f136c66bffce0249eb6d693")  # API Key is in your wandb account, under settings (wandb.ai/settings)

    # Create your wandb run
    run = wandb.init(
        name = "1", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        project = "hw3p2-2", ### Project should be created in your wandb account
        config = config ### Wandb Config for your run
    )

    root = DATA_ROOT  # 数据集的根目录路径




    #train_data = AudioDataset(partition="train-clean-100", use_cmn=True, audio_transformation=transforms) 
    train_data = AudioDataset(partition="train-clean-100", use_cmn=True, audio_transformation=None) 
    val_data = AudioDataset(partition="dev-clean", use_cmn=True, audio_transformation=None)



    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data, 
        num_workers = 4,
        batch_size  = config["batch_size"], 
        pin_memory  = True,                     # 如果使用GPU加速，启用此选项可以加快数据转移到GPU的速度
        persistent_workers = True,
        shuffle     = True,                     # 每个epoch开始时打乱数据集
        collate_fn = train_data.collate_fn      # 使用数据集的collate函数进行批量数据的处理
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data, 
        num_workers = 2,
        batch_size  = config["batch_size"], 
        pin_memory  = True,
        persistent_workers = True,
        shuffle     = False,
        collate_fn = val_data.collate_fn
    )


    print("Batch size: ", config["batch_size"])
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

    model = ASRModel(
        input_size  = 27,
        embed_size  = config['embed_size'],
        output_size = len(PHONEMES)
    ).to(device)
    print(model)

    # # 使用 iter() 和 next() 获取 train_loader 中的第一个批次
    # data_iter = iter(train_loader)
    # x, y, lx, ly = next(data_iter)

    # # 打印形状信息
    # print(x.shape, y.shape, lx.shape, ly.shape)

    # summary(model, 
    #         input_data=[x.to(device), lx.to(device)],
    #         col_names = ("input_size", "output_size", "num_params", "params_percent"),
    #         device=device)

        
    # 定义CTC损失函数作为标准。这个函数用于计算模型输出与目标之间的CTC损失。
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)  
    # 参考CTC损失文档：https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

    # 定义优化器，使用AdamW优化算法更新模型参数，学习率从config中获取。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])  



    decoder = cuda_ctc_decoder(
                          tokens = LABELS,
                          beam_size = config['beam_width'],
                          )
    
    
    # 定义学习率调度器，使用Cosine Annealing调度器逐渐降低学习率。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 如果需要混合精度训练，可以使用GradScaler进行缩放。
    #scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')



    torch.cuda.empty_cache()  # 清空GPU缓存
    gc.collect()  # 进行垃圾回收

    # 开始训练循环
    for epoch in range(config['epochs']):

        print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))

        curr_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

        # 执行训练并获取训练损失
        train_loss = train_model(model, train_loader, criterion, optimizer)
        
        # 执行验证并获取验证损失和验证集上的Levenshtein距离
        valid_loss, valid_dist = validate_model(model, val_loader, decoder, LABELS)
        
        # 根据验证集的Levenshtein距离调整学习率
        scheduler.step()

        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
        print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))

        # 记录训练过程中的指标
        wandb.log({
            'train_loss': train_loss,
            'valid_dist': valid_dist,
            'valid_loss': valid_loss,
            'lr'        : curr_lr
        })

        # 保存当前epoch的模型检查点
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
        #wandb.save(epoch_model_path)
        print("Saved epoch model")

        # 如果当前模型是最佳模型，则保存
        if valid_dist <= best_lev_dist:
            best_lev_dist = valid_dist
            save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
            #wandb.save(best_model_path)
            print("Saved best model")

    run.finish()  # 结束wandb的记录


        