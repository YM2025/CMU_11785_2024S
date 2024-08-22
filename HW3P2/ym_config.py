# 你可以根据需要添加更多的配置项
config = {
    "beam_width": 5,  # 波束宽度，用于解码时的波束搜索算法
    "lr": 2e-3,  
    "epochs": 50,  
    "batch_size": 128,  
    "dropout": 0.2,
    "embed_size": 128, 
}


last_epoch_completed = 0  # 上次完成的训练轮数
start = last_epoch_completed  # 从上次完成的轮次开始
end = config["epochs"]  # 结束轮次，来自配置文件
best_lev_dist = float("inf")  # 如果你从某个检查点重新开始训练，使用当时的最佳 Levenshtein 距离

# 设置检查点模型的保存路径
epoch_model_path = "epoch_checkpoint.pth"  # 这是一个示例路径，可以根据需要更改

# 设置最佳模型的保存路径
best_model_path = "best_model.pth"  # 这是一个示例路径，可以根据需要更改


DATA_ROOT = "E:/cv_data/cmu11785_data/HW3P2_data/11-785-s24-hw3p2/"



# CMU启动笔记本默认的标签设计：
# CMUdict_ARPAbet = {
#     "" : " ",
#     "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
#     "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
#     "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
#     "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
#     "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
#     "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
#     "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
#     "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
#     "[SOS]": "[SOS]", "[EOS]": "[EOS]"
# }

# 我改造了第前2个字符的表示。
CMUdict_ARPAbet = {
    "" : "-",
    "[SIL]": "|", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]


