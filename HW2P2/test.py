
import torch
import torchvision 
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


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)




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

    true_ids = pd.read_csv('/home/nhsh/xiangmu/cmu/datas/11-785-s24-hw2p2-verification/verification_dev.csv')['label'].tolist()

    if mode == 'val':  
        # 以下是用于求最佳阈值和精度的代码
        be_accuracy = 0.0
        be_threshold = 0.0
        for i in range(len(max_similarity_values)):
            threshold = max_similarity_values[i]
            NO_CORRESPONDENCE_LABEL = 'n000000'
            pred_id_strings = []
            for idx, prediction in enumerate(predictions):
                if max_similarity_values[idx] < threshold: # 为什么是<？考虑你的相似度度量方式
                    pred_id_strings.append(NO_CORRESPONDENCE_LABEL)
                else:
                    pred_id_strings.append(known_paths[prediction])

                accuracy = 100 * accuracy_score(pred_id_strings, true_ids)  # 计算验证集的准确率
                
                if accuracy > be_accuracy:
                    be_accuracy = accuracy
                    be_threshold = threshold
        print("Val Acc (Verification) {:.02f}%\t ".format(be_accuracy))
        return be_accuracy,  be_threshold, pred_id_strings
    
    elif mode == 'test': 
        threshold =  # 填上val集计算出的 be_threshold 值。
        NO_CORRESPONDENCE_LABEL = 'n000000'
        pred_id_strings = []
        for idx, prediction in enumerate(predictions):
            if max_similarity_values[idx] < threshold: # 为什么是<？考虑你的相似度度量方式
                pred_id_strings.append(NO_CORRESPONDENCE_LABEL)
            else:
                pred_id_strings.append(known_paths[prediction])
            return pred_id_strings





if __name__ == '__main__':
    

    # 数据预处理、加载
    DATA_DIR    = config['data_dir']
    TRAIN_DIR   = os.path.join(DATA_DIR, "train")
    VAL_DIR     = os.path.join(DATA_DIR, "dev")
    TEST_DIR    = os.path.join(DATA_DIR, "test")
    

    model = ConvNextTinyWithDropout(dropout_prob=0.2, num_classes = 7001).to(DEVICE)


    # 加载模型权重微调并捕获状态信息
    checkpoint = torch.load('/home/ym/HW2P2/checkpoint_verification_copy.pth')
    try:
        load_status = model.load_state_dict(checkpoint['model_state_dict'])
        print(load_status)
    except RuntimeError as e:
        print("加载模型时出现错误:", e)

        
    
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
    
    
    


    eval_verification(unknown_dev_images, known_images,
                                                model, similarity_metric, config['batch_size'], mode='val')
    


