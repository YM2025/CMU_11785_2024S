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



# 定义训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train() # 将模型设置为训练模式
    tloss, correct, total = 0, 0, 0  # 初始化总损失、正确预测数和总样本数
    
    # 初始化进度条
    batch_bar   = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
  
    for frames, phonemes in dataloader:
        frames, phonemes = frames.to(device), phonemes.to(device)  # 将数据移动到指定设备
        
        optimizer.zero_grad()  # 清空梯度
        
        # 使用自动混合精度
        with autocast():
            logits = model(frames)  # 前向传播
            loss = criterion(logits, phonemes)  # 计算损失
        
        scaler.scale(loss).backward()  # 自动缩放梯度（避免f16精度降低导致的溢出或下溢），反向传播
        scaler.step(optimizer)  # 根据缩放的梯度更新模型参数
        scaler.update()  # 动态更新缩放因子
        
        
        tloss += loss.item()  # 累加损失
        pred = torch.argmax(logits, dim=1)  # 预测结果
        correct += (pred == phonemes).sum().item()  # 累加正确预测的数量
        total += phonemes.size(0)  # 累加总样本数
        
        # 更新进度条
        batch_bar.set_postfix(loss="{:.04f}".format(tloss / (len(dataloader))), 
                              acc="{:.04f}%".format(100 * correct / total))
        batch_bar.update()
    
    batch_bar.close()
    
    # 计算平均损失和准确率
    avg_loss = tloss / len(dataloader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc



def train_R_drop(model, dataloader, optimizer, criterion, device, alpha = 0.5):
    model.train()
    tloss, correct, total = 0, 0, 0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for frames, phonemes in dataloader:
        frames, phonemes = frames.to(device), phonemes.to(device)
        
        optimizer.zero_grad()

        # 执行两次前向传播
        with autocast():
            logits1 = model(frames)
            logits2 = model(frames)
            # 计算交叉熵损失
            ce_loss = 0.5 * (criterion(logits1, phonemes) + criterion(logits2, phonemes))
            # 计算 KL 散度损失
            kl_loss = compute_kl_loss(logits1, logits2)
            # 计算总损失
            loss = ce_loss + alpha * kl_loss
        
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tloss += loss.item()
        pred = torch.argmax(logits1, dim=1)  # 使用 logits1 或 logits2 都可
        correct += (pred == phonemes).sum().item()
        total += phonemes.size(0)

        batch_bar.set_postfix(loss="{:.04f}".format(tloss / len(dataloader)), 
                              acc="{:.04f}%".format(100 * correct / total))
        batch_bar.update()

    batch_bar.close()

    # 计算平均损失和准确率
    avg_loss = tloss / len(dataloader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc






# 定义评估函数
def eval(model, dataloader, criterion, device):
    model.eval()  # 将模型设置为评估模式
    vloss, correct, total = 0, 0, 0  # 初始化总损失、正确预测数和总样本数
    
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    
    # 在评估过程中不计算梯度
    with torch.inference_mode():
        for frames, phonemes in dataloader:
            frames, phonemes = frames.to(device), phonemes.to(device)  
            
            logits = model(frames)  # 前向传播
            loss = criterion(logits, phonemes)  # 计算损失

            vloss += loss.item()  # 累加损失
            pred = torch.argmax(logits, dim=1)  # 预测结果
            correct += (pred == phonemes).sum().item()  # 累加正确预测的数量
            total += phonemes.size(0)  # 累加总样本数

            # 更新进度条
            batch_bar.set_postfix(loss="{:.04f}".format(vloss / len(dataloader)), 
                                  acc="{:.04f}%".format(100 * correct / total))
            batch_bar.update()
    
    batch_bar.close()
    
    # 计算平均损失和准确率
    avg_loss = vloss / len(dataloader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc










if __name__ == '__main__':
    
    # WandB 登录和初始化
    wandb.login(key="113991ce0fd655043f136c66bffce0249eb6d693") 
    run = wandb.init(
        name    = "MLP_test_20240606", # 
        reinit  = True, ### Allows reinitalizing runs when you re-run this cell
        #id     = "y28t31uz", ### 如果要恢复之前的运行，请在此处插入特定的运行id
        #resume = "must", ### 你需要它来恢复之前的运行，但在使用它时注释掉reinit = True
        project = "hw1p2-test", ### Project should be created in your wandb account 
        config  = config ### Wandb Config for your run
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    data_path = 'E:/cv_data/cmu11785_data/'
    
    # 开始创建dataset类实例，并将所有数据集进行预处理并缓存在内存中
    train_data = AudioDataset(data_path, 
                              phonemes = PHONEMES, 
                              context=config['context'], 
                              partition= "train-clean-100",
                              use_mask=config['use_mask'], 
                              use_cmn=config['use_cmn'],
                              ) 
    val_data = AudioDataset(data_path, 
                            phonemes = PHONEMES, 
                            context=config['context'], 
                            partition= "dev-clean",
                            use_cmn=config['use_cmn'],
                            ) 
    


    # # 从训练集和验证集中各抽取十分之一的数据
    # subset_indices_train = np.random.choice(len(train_data), len(train_data) // 10, replace=False)
    # subset_indices_val = np.random.choice(len(val_data), len(val_data) // 10, replace=False)
    # train_subset = torch.utils.data.Subset(train_data, subset_indices_train)
    # val_subset = torch.utils.data.Subset(val_data, subset_indices_val)
    
    
    # 初始化数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data, 
        num_workers = 6,
        batch_size  = config['batch_size'], 
        pin_memory  = True,#将batch数据保存在页锁定内存中，加速后续移送GPU
        shuffle     = True, #打乱训练数据可以防止模型学习到数据集中可能存在的任何顺序依赖性，有助于提高模型的泛化能力。
        persistent_workers=False  # 多进程加载数据时，不会重新启动worker进程，从而减少进程开启时间。
    )

    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data, 
        num_workers = 2,
        batch_size  = config['batch_size'],
        pin_memory  = True,
        shuffle     = False, #不打乱数据，因为验证数据集的顺序对模型性能的评估没有影响，保持数据顺序可以帮助更快地加载数据和重复实验。
        persistent_workers=True
    )


    INPUT_SIZE  = (2*config['context'] + 1) * 28  # 1表示MFCC一帧（25ms），28表示每帧的MFCC特征数，2*context+1表示上下文窗口大小
    
    model       = MLP_Net(INPUT_SIZE, len(train_data.phonemes), 
                          config['hidden_sizes'], 
                          config['dropout_rate'], 
                          use_LeakyReLU = config['use_LeakyReLU'], 
                          use_batchnorm = config['use_batchnorm'], 
                          use_dropout = config['use_dropout'], 
                          custom_weight_init = config['custom_weight_init']).to(device)
    
    summary(model, torch.zeros([config['batch_size'], (2*config['context'] + 1) * 28]).to(device)) # 打印模型结构
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr= config['init_lr']) #Defining Optimizer

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 1e-4)

    #清除缓存和垃圾回收
    torch.cuda.empty_cache() # 清空 GPU 缓存
    gc.collect()             # 清理 CPU 内存 (强制进行一次垃圾回收，以确保所有无用的对象都被及时清理，从而释放内存。这主要用于管理 CPU 内存。)


    # 获取当前文件夹路径
    current_dir = os.getcwd() 
    # 构建保存模型的路径
    model_path = os.path.join(current_dir, 'model.pt')

    # 使用wandb监视模型，记录所有指标
    wandb.watch(model, log="all") 

    best_val_acc = 0.0  # 在循环开始前初始化最佳验证准确率


    validation_losses = [] #缓存验证集loss, 用于判定是否开始切换SGD优化器
    use_switch_Adam_to_SGD = config['use_switch_Adam_to_SGD']
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # 获取当前的学习率
        curr_lr = float(optimizer.param_groups[0]['lr'])

        # 训练模型，并返回训练损失和准确率
        if config['use_Rdrop']:
            train_loss, train_acc = train_R_drop(model, train_loader, optimizer, criterion, device)
        else:
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)

        # 更新学习率调度器
        scheduler.step()

        # 对模型进行评估，并返回验证损失和准确率
        val_loss, val_acc = eval(model, val_loader, criterion, device)

        print(f"\tTrain Acc: {train_acc:.04f}%\tTrain Loss: {train_loss:.04f}\tLearning Rate: {curr_lr:.07f}")
        print(f"\tVal Acc: {val_acc:.04f}%\tVal Loss: {val_loss:.04f}")
        
        validation_losses.append(val_loss)
        # 判定是否开始切换优化器
        if use_switch_Adam_to_SGD and should_switch_to_sgd(validation_losses):
            print(f"Switching from Adam to SGD at epoch {epoch + 1}")
            optimizer = torch.optim.SGD(model.parameters(), lr=config['init_lr'], momentum=0.9)
            use_switch_Adam_to_SGD = False

        # 使用wandb记录训练和验证的指标
        wandb.log({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'lr': curr_lr,
        })

        # 仅在验证准确率提升时保存模型
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.04f}% to {val_acc:.04f}%. Saving model...")
            best_val_acc = val_acc  # 更新最佳验证准确率
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'val_acc': val_acc
            }, model_path) 

    # 完成wandb运行
    wandb.finish()
    
