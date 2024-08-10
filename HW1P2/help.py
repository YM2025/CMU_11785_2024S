

# 切换条件，如果验证集损失在连续 3 个epoch没有改善，则切换
def should_switch_to_sgd(validation_losses, threshold=4):
    if len(validation_losses) >= threshold:
        # 检查最近几个epoch的损失是否持平或上升
        if all(validation_losses[i] <= validation_losses[i + 1] for i in range(-threshold, -1)):
            return True
    return False




# import torch.nn.functional as F
# # 定义计算KL散度损失的函数
# def compute_kl_loss(p, q, pad_mask=None):
#     p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
#     q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
#     if pad_mask is not None:
#         p_loss.masked_fill_(pad_mask, 0.)
#         q_loss.masked_fill_(pad_mask, 0.)

#     p_loss = p_loss.sum()
#     q_loss = q_loss.sum()

#     loss = (p_loss + q_loss) / 2
#     return loss

import torch.nn.functional as F
import torch
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




