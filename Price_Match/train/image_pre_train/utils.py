import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed, cuda_enable):
    random.seed(seed)
    np.random.seed(seed)
    if cuda_enable and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def get_device(cuda_enable):
    if cuda_enable:
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def JS_loss(img_c, pos_img_c_list, neg_img_c_list):
    pos = [torch.sum(torch.mul(img_c, item), dim=1) for item in pos_img_c_list]
    neg = [torch.sum(torch.mul(img_c, item)*(-1), dim=1) for item in neg_img_c_list]

    loss = torch.tensor(0, dtype=torch.float)
    for item in pos:
        loss = loss - torch.sum(F.logsigmoid(item))
    for item in neg:
        loss = loss - torch.sum(F.logsigmoid(item))
    return loss