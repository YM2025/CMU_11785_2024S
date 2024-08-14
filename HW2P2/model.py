

config = {
    'data_dir'      : "/home/nhsh/xiangmu/cmu/datas",
    'batch_size': 128, 
    'lr': 1e-3,
    'epochs': 15,
}


    
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

# convnext-t 模型加个dropout层
class ConvNextTinyWithDropout(nn.Module):
    def __init__(self, dropout_prob, num_classes):
        super(ConvNextTinyWithDropout, self).__init__()

        #self.model = convnext_tiny(weights=None)
        self.model = convnext_tiny(weights="DEFAULT") #下载并使用convnext-t官方预训练模型


        # 获取原始分类器的各层
        original_classifier = self.model.classifier
        
        # 获取最后的分类层的输入特征数
        num_ftrs = self.model.classifier[2].in_features
        
        # 重定义分类器，插入 Dropout 层
        self.model.classifier = nn.Sequential(
            original_classifier[0],  # LayerNorm
            original_classifier[1],  # Flatten
            nn.Dropout(p=dropout_prob),  # 新增 Dropout 层
            nn.Linear(num_ftrs, num_classes),  # 全连接层，使用其输入特征数和输出类别数
        )

    def forward(self, x, return_feats=False, return_all=False):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        
        if return_all: # 两个都返回，用于损失函数=交叉熵损失+arcface损失融合。
            x1 = x.view(x.size(0), -1)
            x2 = self.model.classifier(x)
            return x1, x2
        
        if return_feats:
            x = x.view(x.size(0), -1)# 拉平张量，保持批量大小不变
            return x
        else:
            x = self.model.classifier(x)
            return x







import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from collections import OrderedDict
import torch.nn.functional as F

# DropBlock结构
class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device, dtype=x.dtype) < gamma).float()
        block_mask = 1 - F.max_pool2d(mask[:, None, :, :], kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        block_mask = block_mask.repeat(1, x.shape[1], 1, 1).view_as(x)
        return x * block_mask * (block_mask.numel() / (1e-7 + block_mask.sum()))

class ModifiedConvNeXtTiny(nn.Module):
    def __init__(self, block_size, keep_prob, num_classes=500):
        super(ModifiedConvNeXtTiny, self).__init__()
        self.base_model = convnext_tiny(weights=None)
        
        # 修改最后一层输出类别数
        self.base_model.classifier[2] = nn.Linear(self.base_model.classifier[2].in_features, num_classes)
        
        # 添加 DropBlock 到前两个阶段的第一个卷积层
        for i in range(2):  # 前两个stage
            original_stage = self.base_model.features[i]
            modified_stage = []
            dropblock_added = False
            for layer in original_stage:
                modified_stage.append(layer)
                if isinstance(layer, nn.Conv2d) and not dropblock_added:
                    # 在第一个Conv2D后添加DropBlock，并确保只添加一次
                    modified_stage.append(DropBlock(block_size=block_size, keep_prob=keep_prob))
                    dropblock_added = True
            self.base_model.features[i] = nn.Sequential(*modified_stage)
            
    def forward(self, x):
        x = self.base_model(x)  # Process through features
        return x




# convnext_tiny网络中添加DropBlock + dropout结构
class DropBlock_dropout_ConvNeXtTiny(nn.Module):
    def __init__(self, block_size, keep_prob, dropout_prob=0.2, num_classes=500):
        super(DropBlock_dropout_ConvNeXtTiny, self).__init__()
        
        self.model = convnext_tiny(weights="DEFAULT")
        
        # 添加 DropBlock 到前两个阶段的第一个卷积层
        for i in range(2):  # 前两个stage
            original_stage = self.model.features[i]
            modified_stage = []
            dropblock_added = False
            for layer in original_stage:
                modified_stage.append(layer)
                if isinstance(layer, nn.Conv2d) and not dropblock_added:
                    # 在第一个Conv2D后添加DropBlock，并确保只添加一次
                    modified_stage.append(DropBlock(block_size=block_size, keep_prob=keep_prob))
                    dropblock_added = True
            self.model.features[i] = nn.Sequential(*modified_stage)
        

        # 获取原始分类器的各层
        original_classifier = self.model.classifier
        
        # 获取最后的分类层的输入特征数
        num_ftrs = self.model.classifier[2].in_features
        
        # 重定义分类器，插入 Dropout 层
        self.model.classifier = nn.Sequential(
            original_classifier[0],  # LayerNorm
            original_classifier[1],  # Flatten
            nn.Dropout(p=dropout_prob),  # 新增 Dropout 层
            nn.Linear(num_ftrs, num_classes),  # 全连接层，使用其输入特征数和输出类别数
        )
        
    def forward(self, x, return_feats=False):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        if return_feats:
            return x
        else:
            x = self.model.classifier(x)
            return x