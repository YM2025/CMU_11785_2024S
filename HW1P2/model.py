
import torch
import torch.nn as nn
import numpy as np


config = {
    'epochs'        : 10,
    'batch_size'    : 2048,
    'context'       : 30,
    'init_lr'       : 1e-3,
    'dropout_rate'  : 0.2,
    'hidden_sizes'  : np.array([1,2,4,4,2,1]) * 512,
    'use_LeakyReLU' : False,
    'use_batchnorm' : True,
    'use_dropout'   : True,
    'custom_weight_init' : False,
    'use_cmn'       : True,
    'use_mask'      : 0, 
    'use_switch_Adam_to_SGD': False,
    'use_Rdrop'     : False,
}
# use_cmn：是否对数据进行 cepstral mean normalization 归一化。

# use_mask备注：
# 0：不使用mask
# 1：dropout式mask：所有特征点，独立5%几率置为0.
# 2：时间掩码：10%的随机位置帧置为0.
# 3：频率掩码：10%的随机行的特征置为0.
# 4：时间+频率掩码：2+3同时应用。各5%概率



class MLP_Net(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_rate, use_LeakyReLU = False, use_batchnorm=True, use_dropout=True, custom_weight_init=False):
        super(MLP_Net, self).__init__()
        #self.layers = nn.ModuleList()  # 使用 ModuleList 以确保所有层都被注册
        self.layers = []
        for hs in hidden_sizes:
            self.layers.extend(self._mlp_layer_provider(input_size, hs, dropout_rate, use_LeakyReLU, use_batchnorm, use_dropout))
            input_size = hs
        self.layers.append(nn.Linear(input_size, output_size))  # 输出层
        self.model = nn.Sequential(*self.layers)
        
        if custom_weight_init:
            self.model.apply(self._weights_init)  # 采用自定义权重初始化方法。如不指定，PyTorch 默认用 Kaiming 均匀初始化方法。
    
    
    def forward(self, x):
        return self.model(x)
    
    
    def _mlp_layer_provider(self, input_size, hidden_size, dropout_rate, use_LeakyReLU, use_batchnorm, use_dropout):
        
        modules = [nn.Linear(input_size, hidden_size)]
        
        # 记得放激励函数前面
        if use_batchnorm: 
            modules.append(nn.BatchNorm1d(hidden_size))
        
        if use_LeakyReLU: 
            modules.append(nn.LeakyReLU()) 
        else: 
            modules.append(nn.GELU())
        
        if use_dropout: 
            modules.append(nn.Dropout(dropout_rate))
            
        return modules

        
    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            # 随机正态分布初始化
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)




