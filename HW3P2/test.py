import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from ym_config import *
from ym_dataset import *
from ym_model import *
from ym_helps import *
from torchaudio.models.decoder import cuda_ctc_decoder

# 设置设备为GPU或CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# 加载最佳模型
model = ASRModel(
    input_size=27,
    embed_size=config['embed_size'],
    output_size=len(PHONEMES)
).to(device)


# 加载checkpoint并提取模型的状态字典
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 设置模型为评估模式


# 加载验证数据集
val_data = AudioDataset(partition="dev-clean", use_cmn=True, audio_transformation=None)
val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    num_workers=1,
    batch_size=config["batch_size"],
    pin_memory=True,
    persistent_workers=True,
    shuffle=False,
    collate_fn=val_data.collate_fn
)

# 定义CTC损失函数
criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

# 初始化解码器
decoder = cuda_ctc_decoder(
    tokens=LABELS,
    beam_size=config['beam_width'],
)

# 评估模型
def evaluate_model(model, val_loader, decoder, phoneme_map=LABELS):
    model.eval()  # 设置模型为评估模式
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0  # 初始化总损失
    vdist = 0  # 初始化总的Levenshtein距离

    for i, data in enumerate(val_loader):
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
    torch.cuda.empty_cache()  # 清空GPU缓存
    gc.collect()  # 进行垃圾回收

    # 执行验证并获取验证损失和验证集上的Levenshtein距离
    valid_loss, valid_dist = evaluate_model(model, val_loader, decoder, LABELS)

    print("Validation Loss: {:.04f}".format(valid_loss))
    print("Validation Levenshtein Distance: {:.04f}".format(valid_dist))
