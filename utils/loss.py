import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # 防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(prob, labels):
    """
    prob:  (N, C) 已softmax的概率
    labels: (N,) 类别索引
    """
    # 防止 log(0)
    eps = 1e-9
    return -np.mean(np.log(prob[np.arange(len(labels)), labels] + eps)) # 只计算正确的类别那一项

def mse_loss(x, y):
    """
    x, y: (N, D)
    """
    return np.mean((x - y) ** 2)

def bce_loss(pred, target):
    """
    pred: (N,) 预测概率
    target: (N,) 0或1标签
    """
    eps = 1e-9
    pred = np.clip(pred, eps, 1 - eps) # 防止 log(0)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
