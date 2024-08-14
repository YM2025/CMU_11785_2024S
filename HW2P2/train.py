import os
# 设置正确的工作目录
desired_path = '/home/ym/HW2P2'
os.chdir(desired_path)
print("当前工作目录:", os.getcwd())

import torch
import torchvision # This library is used for image-based operations (Augmentations)
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import wandb
import matplotlib.pyplot as plt
import random

from model import *
from helps import *


# 定义所有库的随机种子。
SEED = 11785
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)




# 交叉熵损失函数版
def train(model, dataloader, optimizer, criterion):
    #scaler = torch.cuda.amp.GradScaler() # 初始化梯度缩放器，用于自动混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()

    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss  = 0

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # # 清空梯度

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        #with torch.cuda.amp.autocast(): # This implements mixed precision. Thats it!
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 使用新的autocast初始化方式
            outputs = model(images)
            loss    = criterion(outputs, labels)

        num_correct     += int((torch.argmax(outputs, axis=1) == labels).sum()) # 累计 batch 中正确预测的个数
        total_loss      += float(loss.item())

        scaler.scale(loss).backward()  # 自动缩放梯度（避免f16精度降低导致的溢出或下溢），并反向传播
        scaler.step(optimizer) # # 根据缩放的梯度更新模型参数
        scaler.update() # 动态更新缩放因子

        # 更新进度条
        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss        = "{:.04f}".format(float(total_loss / (i + 1))),
            num_correct = num_correct,
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update() 

    batch_bar.close() # You need this to close the tqdm bar

    acc         = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss  = float(total_loss / len(dataloader))

    return acc, total_loss


# arcface损失函数版
def train_arcface(model, dataloader, optimizer, criterion):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss  = 0

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # # 清空梯度

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  
            feature = model(images, return_feats=True)
            outputs = metric_arcface(feature, labels)
            loss = criterion(outputs, labels)

        num_correct     += int((torch.argmax(outputs, axis=1) == labels).sum()) # 累计 batch 中正确预测的个数
        total_loss      += float(loss.item())

        scaler.scale(loss).backward()  # 自动缩放梯度（避免f16精度降低导致的溢出或下溢），并反向传播
        scaler.step(optimizer) # # 根据缩放的梯度更新模型参数
        scaler.update() # 动态更新缩放因子

        # 更新进度条
        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss        = "{:.04f}".format(float(total_loss / (i + 1))),
            num_correct = num_correct,
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update() 

    batch_bar.close() # You need this to close the tqdm bar

    acc         = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss  = float(total_loss / len(dataloader))

    return acc, total_loss



# 交叉熵损失函数版 + Rdrop
def train_R_drop(model, dataloader, optimizer, criterion):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss  = 0

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # # 清空梯度

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 执行两次前向传播
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 使用新的autocast初始化方式
            outputs1 = model(images)
            outputs2 = model(images)
            
            # 计算交叉熵损失
            ce_loss = 0.5 * (criterion(outputs1, labels) + criterion(outputs2, labels))
            # 计算 KL 散度损失
            kl_loss = compute_kl_loss(outputs1, outputs2)
            # 计算总损失
            alpha = 1 #这是个权重超参数（HW1P2里，1感觉是最好的）
            loss = ce_loss + alpha * kl_loss

        num_correct     += int((torch.argmax(outputs1, axis=1) == labels).sum()) # 累计 batch 中正确预测的个数
        total_loss      += float(loss.item())

        scaler.scale(loss).backward()  # 自动缩放梯度（避免f16精度降低导致的溢出或下溢），并反向传播
        scaler.step(optimizer) # # 根据缩放的梯度更新模型参数
        scaler.update() # 动态更新缩放因子

        # 更新进度条
        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss        = "{:.04f}".format(float(total_loss / (i + 1))),
            num_correct = num_correct,
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update() 

    batch_bar.close() # You need this to close the tqdm bar

    acc         = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss  = float(total_loss / len(dataloader))

    return acc, total_loss



# 备注：主要用于微调，将训练好的非arcface版模型，继续训练10个epoch，损失函数=交叉熵损失+arcface损失。
def train_warmup_arcface(model, dataloader, optimizer, criterion):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss  = 0

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # # 清空梯度

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 使用新的autocast初始化方式
            feature, out2 = model(images, return_all=True)
            out1 = metric_arcface(feature, labels)
            loss = 0.5*criterion(out1, labels) + 0.5*criterion(out2, labels)

        num_correct     += int((torch.argmax(out2, axis=1) == labels).sum()) # 累计 batch 中正确预测的个数
        total_loss      += float(loss.item())

        scaler.scale(loss).backward()  # 自动缩放梯度（避免f16精度降低导致的溢出或下溢），并反向传播
        scaler.step(optimizer) # # 根据缩放的梯度更新模型参数
        scaler.update() # 动态更新缩放因子

        # 更新进度条
        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss        = "{:.04f}".format(float(total_loss / (i + 1))),
            num_correct = num_correct,
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update() 

    batch_bar.close() # You need this to close the tqdm bar

    acc         = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss  = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, dataloader, criterion, use_arcface=False):

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Get model outputs
        with torch.inference_mode():
            if use_arcface: #判断是否使用了arcface
                features = model(images, return_feats=True)  # 获取特征向量
                outputs = metric_arcface(features, labels)  # 计算ArcFace输出
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()

    batch_bar.close()
    acc = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss




def eval_verification(unknown_images, known_images, model, similarity, batch_size= config['batch_size'], mode='val'):
    # unknown_images: 未知身份的图像数据集
    # known_images: 已知身份的图像数据集
    # model: 用于提取特征的深度学习模型
    # similarity: 计算相似度的函数
    # batch_size: 每次处理的图像批次大小，默认为配置文件中的batch_size
    # mode: 模式，可以是'val'（验证模式）或'test'（测试模式）

    unknown_feats, known_feats = [], []  # 用于存储未知和已知图像的特征向量

    # 初始化进度条，用于显示处理进度
    batch_bar = tqdm(total=len(unknown_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
    model.eval()  # 设置模型为评估模式，关闭dropout和batch normalization

    # 分批处理未知图像，避免内存不足（OOM）错误
    for i in range(0, unknown_images.shape[0], batch_size):
        unknown_batch = unknown_images[i:i+batch_size]  # 获取当前批次的图像数据

        with torch.no_grad():  # 关闭梯度计算，减少内存占用
            unknown_feat = model(unknown_batch.float().to(DEVICE), return_feats=True)  # 提取当前批次的图像特征
        unknown_feats.append(unknown_feat)  # 将特征添加到未知特征列表中
        batch_bar.update()  # 更新进度条

    batch_bar.close()  # 关闭进度条

    # 重新初始化进度条，用于处理已知图像
    batch_bar = tqdm(total=len(known_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)

    for i in range(0, known_images.shape[0], batch_size):
        known_batch = known_images[i:i+batch_size]  # 获取当前批次的图像数据
        with torch.no_grad():
              known_feat = model(known_batch.float().to(DEVICE), return_feats=True)  # 提取当前批次的图像特征

        known_feats.append(known_feat)  # 将特征添加到已知特征列表中
        batch_bar.update()  # 更新进度条

    batch_bar.close()  # 关闭进度条

    # 将所有批次的特征向量进行拼接
    unknown_feats = torch.cat(unknown_feats, dim=0)
    known_feats = torch.cat(known_feats, dim=0)

    # 计算每个未知图像特征与所有已知图像特征之间的相似度
    # similarity_values 是一个二维张量（矩阵），它的形状是 (num_known_images, num_unknown_images)
    # 矩阵中的每个元素表示一个已知图像与一个未知图像之间的相似度。
    similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])

    # 获取最大相似度值 和 对应的已知图像索引（预测结果）
    # max(0) 中的 0 表示在第0维度上取最大值，也就是对矩阵的每一列进行操作。这里的“列”代表每个未知图像与所有已知图像的相似度。
    max_similarity_values, predictions = similarity_values.max(0)
    max_similarity_values, predictions = max_similarity_values.cpu().numpy(), predictions.cpu().numpy() #结果转换为 NumPy 数组

    # 使用.squeeze()方法将predictions降维到一维数组
    predictions = predictions.squeeze()
    
    # 注意，在未知身份中，有些身份在已知身份中没有对应的匹配
    # 因此，这些身份与所有已知身份都不应该相似，即最大相似度应低于某个阈值
    # 对于早期提交，你可以忽略没有对应身份的情况，简单地选择相似度最大的身份
    pred_id_strings = [known_paths[i] for i in predictions]  # 将预测的索引映射为身份字符串
    

    if mode == 'val':  # 验证模式
        true_ids = pd.read_csv('/home/nhsh/xiangmu/cmu/datas/11-785-s24-hw2p2-verification/verification_dev.csv')['label'].tolist()
        accuracy = 100 * accuracy_score(pred_id_strings, true_ids)  # 计算验证集的准确率
        return accuracy, pred_id_strings

    elif mode == 'test':  # 测试模式
        return pred_id_strings





if __name__ == '__main__':
    
    # 1）wandb配置
    wandb.login(key="113991ce0fd655043f136c66bffce0249eb6d693")  # API Key is in your wandb account, under settings (wandb.ai/settings)

    # Create your wandb run
    run = wandb.init(
        name = "arcface", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        project = "hw2p2-3080-arcface", ### Project should be created in your wandb account
        config = config ### Wandb Config for your run
    )

    # 2）数据预处理、加载
    DATA_DIR    = config['data_dir']
    TRAIN_DIR   = os.path.join(DATA_DIR, "train")
    VAL_DIR     = os.path.join(DATA_DIR, "dev")
    TEST_DIR    = os.path.join(DATA_DIR, "test")
    
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 空间变换
            torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 透视变换
            torchvision.transforms.GaussianBlur(kernel_size=1),  # 模糊
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色调整
            torchvision.transforms.ToTensor(),  # 转换为张量
            torchvision.transforms.Normalize(mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587]),  # 归一化
            torchvision.transforms.RandomErasing(scale=(0.05, 0.05))  # 随机擦除
        ]) 

    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587]),
        ]) 


    train_dataset   = torchvision.datasets.ImageFolder(TRAIN_DIR, transform = train_transforms)
    val_dataset   = torchvision.datasets.ImageFolder(VAL_DIR, transform = valid_transforms)
    
    
    # # ———————————————————————————————— 调参测试 ————————————————————————————————
    # # 选择前500个类别
    # selected_classes = train_dataset.classes[:500]
    # class_to_idx = {cls_name: idx for idx, cls_name in enumerate(selected_classes)}

    # # 过滤数据集，保留属于所选类别的样本
    # def filter_indices(dataset, selected_classes):
    #     return [i for i, (path, target) in enumerate(dataset.samples) if dataset.classes[target] in selected_classes]

    # train_indices = filter_indices(train_dataset, selected_classes)
    # val_indices = filter_indices(val_dataset, selected_classes)

    # # 创建新的数据子集
    # train_dataset_500 = torch.utils.data.Subset(train_dataset, train_indices)
    # val_dataset_500 = torch.utils.data.Subset(val_dataset, val_indices)

    # # 更新类别和类索引
    # for ds in [train_dataset_500, val_dataset_500]:
    #     ds.dataset.classes = selected_classes
    #     ds.dataset.class_to_idx = class_to_idx
    # # ———————————————————————————————— 调参测试 ————————————————————————————————    
    
    
    train_loader = torch.utils.data.DataLoader(dataset       = train_dataset,
                                                batch_size    = config['batch_size'],
                                                shuffle        = True,
                                                num_workers = 6, 
                                                pin_memory    = True,
                                                persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(dataset       = val_dataset,
                                                batch_size    = config['batch_size'],
                                                shuffle        = False,
                                                num_workers = 2,
                                                persistent_workers=True,
                                                pin_memory    = True,
                                                )
    
    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", config['batch_size'])
    print("Train batches        : ", train_loader.__len__())
    print("Val batches          : ", valid_loader.__len__())
    
    # print("Number of classes    : ", len(selected_classes))
    # print("No. of train images  : ", len(train_dataset_500))
    # print("No. of val images    : ", len(val_dataset_500))
    # print("Shape of train image : ", train_dataset_500[0][0].shape)
    # print("Batch size           : ", config['batch_size'])
    # print("Train batches        : ", len(train_loader))
    # print("Val batches          : ", len(valid_loader))



    
    # 3）初始化模型、损失函数、优化器、学习器、
    # model = Network().to(DEVICE)
    from torchvision.models import convnext_tiny
    import torch.nn as nn
    # convnext基本结构：
    # ConvNeXt(
    #     (stem): Sequential(...)
    #     (stages): Sequential(...)
    #     (classifier): Sequential(
    #            (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
    #            (1): Flatten(start_dim=1, end_dim=-1)
    #            (2): Linear(in_features=768, out_features=500, bias=True)
    #     )
    # )
    
    # ## 1 原版 convnext
    # model = convnext_tiny(weights="DEFAULT").to(DEVICE)
    # # model.classifier[2]：模型 classifier 子模块的第三个层。
    # model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(selected_classes)).to(DEVICE)


    ##2 增加 dropout 版 convnext
    model = ConvNextTinyWithDropout(dropout_prob=0.2, num_classes = len(train_dataset.classes)).to(DEVICE)

    # # 3 增加 DropBlock 版 convnext
    # model = ModifiedConvNeXtTiny(block_size=7, keep_prob=0.8, num_classes=len(selected_classes)).to(DEVICE)
    # print(model)
    
    
    ## 4 dropBlock + droput 版
    #model = DropBlock_dropout_ConvNeXtTiny(block_size=7, keep_prob=0.8, dropout_prob=0.2, num_classes=len(selected_classes)).to(DEVICE)
            
    
    #summary(model, (3, 224, 224))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing= 0.1) 
    #criterion = FocalLoss(gamma=2, label_smooth=0.1) #使用FocalLoss + CrossEntropyLoss + label_smoothing
    
    metric_arcface = ArcMarginProduct(768, len(train_dataset.classes), s=30, m=0.5, easy_margin=False)
    # metric_arcface = ArcMarginProduct(768, len(train_dataset.classes), s=10, m=0.2, easy_margin=False)
    metric_arcface.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    
    
    
    ############ 学习率预热 ############
    # 定义学习率预热的lambda函数
    def warmup_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5 
        return 1.0
    
    # 合并两个调度器
    class CombinedScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_scheduler, cosine_scheduler, warmup_epochs = 5 ):
            self.warmup_scheduler = warmup_scheduler
            self.cosine_scheduler = cosine_scheduler
            self.warmup_epochs = warmup_epochs
            super().__init__(optimizer, last_epoch=-1)  # 确保使用默认的last_epoch

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return self.warmup_scheduler.get_last_lr()
            else:
                self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                return self.cosine_scheduler.get_last_lr()

        def step(self, epoch=None):
            # 手动设置epoch
            if epoch is None:
                self.last_epoch += 1
                epoch = self.last_epoch
            else:
                self.last_epoch = epoch
            
            # 根据当前的epoch调用适当的scheduler
            if self.last_epoch < self.warmup_epochs:
                self.warmup_scheduler.step(epoch)
            else:
                self.cosine_scheduler.step(epoch - self.warmup_epochs)
    
    # 创建LambdaLR调度器用于预热
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # 创建余弦退火调度器
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 45)

    # 初始化合并调度器
    scheduler = CombinedScheduler(optimizer, warmup_scheduler, cosine_scheduler)
    

    # # 加载模型权重微调并捕获状态信息
    # checkpoint = torch.load('/home/ym/HW2P2/checkpoint_verification_copy.pth')
    # try:
    #     load_status = model.load_state_dict(checkpoint['model_state_dict'])
    #     print(load_status)
    # except RuntimeError as e:
    #     print("加载模型时出现错误:", e)

        
    
    # 获取已知身份文件夹的路径列表
    # 使用glob模式匹配来查找所有已知的身份目录
    known_regex = os.path.join(DATA_DIR, "11-785-s24-hw2p2-verification/known/*/*")
    known_paths = [i.split(os.path.sep)[-2] for i in sorted(glob.glob(known_regex))] # 形如：[n000006, , , n012345]

    # 获取未知开发集和测试集的图像路径列表
    unknown_dev_regex = os.path.join(DATA_DIR, "11-785-s24-hw2p2-verification/unknown_dev/*")
    unknown_test_regex = os.path.join(DATA_DIR, "11-785-s24-hw2p2-verification/unknown_test/*")

    # 使用PIL和tqdm加载已知和未知的图像，tqdm用于显示加载进度
    unknown_dev_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_dev_regex)))]
    unknown_test_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_test_regex)))]
    known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]

    # 图像转换设置，这里只使用了ToTensor()转换
    # ToTensor()转换将PIL图像或NumPy ndarray转换为FloatTensor，并缩放像素值范围到[0, 1]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587])]
        )

    # 应用转换并将列表中的图像堆叠成一个张量
    # torch.stack()创建一个新的维度并将图像张量堆叠在这个维度上
    unknown_dev_images = torch.stack([transforms(x) for x in unknown_dev_images]) # torch.Size([360, 3, 224, 224])
    unknown_test_images = torch.stack([transforms(x) for x in unknown_test_images]) # torch.Size([720, 3, 224, 224])
    known_images  = torch.stack([transforms(y) for y in known_images ]) # torch.Size([960, 3, 224, 224])
    
    # 打印张量形状以帮助理解数据结构
    print("Unknown Dev Images Shape:", unknown_dev_images.shape)
    print("Unknown Test Images Shape:", unknown_test_images.shape)
    print("Known Images Shape:", known_images.shape)

    # 定义相似度度量标准，这里使用了余弦相似度
    # dim=1表示在特征维度上计算相似度，eps是一个小值，用于避免除零错误
    similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6)
    
    
    #清除缓存和垃圾回收
    torch.cuda.empty_cache() # 清空 GPU 缓存
    gc.collect()             # 清理 CPU 内存 (强制进行一次垃圾回收，以确保所有无用的对象都被及时清理，从而释放内存。这主要用于管理 CPU 内存。)
    
    
    
    best_class_acc      = 0.0
    best_ver_acc        = 0.0

    for epoch in range(config['epochs']):

        # print("\nEpoch {}/{}".format(epoch+1, config['epochs']))
        curr_lr = float(optimizer.param_groups[0]['lr'])


        #train_acc, train_loss = train(model, train_loader, optimizer, criterion)
        train_acc, train_loss = train_R_drop(model, train_loader, optimizer, criterion)
        #train_acc, train_loss = train_arcface(model, train_loader, optimizer, criterion)
        #train_acc, train_loss = train_warmup_arcface(model, train_loader, optimizer, criterion)
        
        # # 更新学习率调度器
        scheduler.step() 
        

        
        print("\nEpoch {}/{}: \nTrain Acc (Classification) {:.04f}%\t Train Loss (Classification) {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, config['epochs'], train_acc, train_loss, curr_lr))

        # 打开use_arcface来使用arcface版
        val_acc, val_loss = validate(model, valid_loader, criterion, use_arcface = False) 
        
        
        print("Val Acc (Classification) {:.02f}%\t Val Loss (Classification) {:.04f}".format(val_acc, val_loss))
        
        ver_acc, pred_id_strings = eval_verification(unknown_dev_images, known_images,
                                                    model, similarity_metric, config['batch_size'], mode='val')
        print("Val Acc (Verification) {:.02f}%\t ".format(ver_acc))
        
        
        wandb.log({"train_classification_acc": train_acc,
                "train_classification_loss":train_loss,
                "val_classification_acc": val_acc,
                "val_classification_loss": val_loss,
                "val_verification_acc": ver_acc,
                "learning_rate": curr_lr})


        if val_acc >= best_class_acc:
            best_valid_acc = val_acc
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, './checkpoint_classification.pth')
            #wandb.save('checkpoint_verification.pth')
            print("Saved best classification model")

        if ver_acc >= best_ver_acc:
            best_ver_acc = ver_acc
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': ver_acc,
                        'epoch': epoch}, './checkpoint_verification.pth')
            #wandb.save('checkpoint_verification.pth')
            print("Saved verification model")

    ## Finish your wandb run
    run.finish()