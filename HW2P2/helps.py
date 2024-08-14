
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np


# 定义计算KL散度损失的函数
def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean') # 必须要设置为 batchmean，否则损失很大。
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')

    # 如果有掩码，则应用掩码
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(-1)  # 确保掩码的维度与p_loss和q_loss匹配
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # 计算最终的平均损失
    loss = (p_loss + q_loss) / 2
    return loss



# 交叉熵损失+focalloss损失
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, label_smooth=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        if label_smooth == 0: 
            self.ce = torch.nn.CrossEntropyLoss() 
        else:
            self.ce = torch.nn.CrossEntropyLoss(label_smoothing = label_smooth)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()





# arcface损失函数
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出类别数量
        self.s = s  # 放大系数
        self.m = m  # 角度边距
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # 分类层权重参数
        nn.init.xavier_uniform_(self.weight)  # 用Xavier均匀分布初始化权重

        self.easy_margin = easy_margin  # 是否使用easy margin策略
        self.cos_m = math.cos(m)  # 预计算角度边距的余弦值
        self.sin_m = math.sin(m)  # 预计算角度边距的正弦值
        self.th = math.cos(math.pi - m)  # 阈值用于判断是否使用边距
        self.mm = math.sin(math.pi - m) * m  # 辅助计算的常量

    def forward(self, input, label):
        # --------------------------- cos(theta) 和 phi(theta) 的计算 ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # 计算输入特征与权重的归一化余弦相似度
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))  # 计算余弦值对应的正弦值
        phi = cosine * self.cos_m - sine * self.sin_m  # 计算 phi(theta) 的值
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)  # 使用 easy margin 策略
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # 判断是否需要应用边距

        # --------------------------- 将标签转换为 one-hot 编码 ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')  # 创建一个全零张量
        one_hot = torch.zeros(cosine.size(), device='cuda')  # 创建一个全零张量
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 将标签转为 one-hot 编码

        # -------------通过torch.where生成输出 -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # 根据标签选择使用 phi 或者 cosine
        output *= self.s  # 放大输出

        return output  # 返回最终的输出

