import numpy as np

def log_softmax(x, axis):
    ret = x - np.max(x, axis=axis, keepdims=True)
    lsm = np.log(np.sum(np.exp(ret), axis=axis, keepdims=True))
    return ret - lsm


def array_to_str(arr, vocab):
    return " ".join(vocab[a] for a in arr)


def get_prediction_nll(out, targ):
    out = log_softmax(out, 1)
    nlls = out[np.arange(out.shape[0]), targ] # 提取 out 中所有样本在 targ 所指定类别索引上的对数概率。
    nll = -np.mean(nlls)
    return nll

def make_generation_text(inp, pred, vocab):
    outputs = u""
    for i in range(inp.shape[0]):
        w1 = array_to_str(inp[i], vocab)
        w2 = array_to_str(pred[i], vocab)
        outputs += u"Input | Output #{}: {} | {}\n".format(i, w1, w2)
    return outputs