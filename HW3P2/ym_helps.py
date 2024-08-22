
import Levenshtein  # 导入Levenshtein模块，用于计算Levenshtein距离
import torch

from ym_config import *
from ym_dataset import *
from ym_model import * 



# 计算预测字符串与实际标签字符串之间的Levenshtein距离。
# Levenshtein距离衡量两个字符串之间的编辑距离（插入、删除、替换操作的最小次数），并返回平均距离作为损失的一种衡量。
def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS):
    
    dist = 0
    batch_size = label.shape[0]
    
    logits_len = output_lens.to('cuda')
    decoded_text = decoder(output, logits_len.to(torch.int32))  # 解码
    
    for i in range(batch_size):
        # 获取批次中每个元素的预测字符串和标签字符串
        pred_string = ''.join([c for c in decoded_text[i][0].words])  # 获取第 i 个样本预测的字符串
        label_string = ''.join([PHONEME_MAP[n] for n in label[i][:label_lens[i]]])  # 获取第 i 个标签字符串
        dist += Levenshtein.distance(pred_string, label_string)  # 计算 Levenshtein 距离并累加

    dist /= batch_size  # 计算平均 Levenshtein 距离
    
    return dist  # 返回平均距离









def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {
            'model_state_dict': model.state_dict(),  # 保存模型的状态字典
            'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的状态字典
            'scheduler_state_dict': scheduler.state_dict(),  # 保存学习率调度器的状态字典
            metric[0]: metric[1],  # 保存特定指标的值，如验证集准确率
            'epoch': epoch  # 保存当前训练的轮次
        },
        path  # 指定保存的路径
    )


def load_model(path, model, metric='valid_acc', optimizer=None, scheduler=None):
    
    checkpoint = torch.load(path)  # 加载模型检查点
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态字典

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态字典（如果提供）
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 加载调度器状态字典（如果提供）

    epoch = checkpoint['epoch']  # 获取保存的轮次
    metric = checkpoint[metric]  # 获取保存的特定指标值

    return [model, optimizer, scheduler, epoch, metric]  # 返回模型、优化器、调度器、轮次和指标值
