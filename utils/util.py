import os
import sys
import math
import random
import logging
import datetime
import numpy as np

import torch

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True # 是否追求训练速度，不追求可复现性
    torch.backends.cudnn.deterministic = False # 保证结果可复现

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr_ratio = epoch / args.warmup_epochs
    else:
        if args.lr_decay == 'cosine_decay':
            lr_ratio = 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.lr_decay == 'multi_step':
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            lr_ratio = (args.lr_decay_rate ** steps)
        elif args.lr_decay == 'constant':
            lr_ratio = 1.0
        else:
            lr_ratio = 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_ratio
    return args.lr * lr_ratio

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_logger(name, save_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fh = logging.FileHandler(os.path.join(save_dir, f"{timestamp}.log"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger