import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import timm
import torch
from torch import nn
import torch.nn.functional as F

import engine
from dataset import ShopeeDataset
from custom_scheduler import ShopeeScheduler
from augmentations import get_train_transforms
from argmargin import ArcMarginProduct
from model import ShopeeModel

DATA_DIR = '../input/shopee-product-matching/train_images'
TRAIN_CSV = '../input/utils-shopee/folds.csv'
MODEL_PATH = './'


class CFG:
    seed = 54
    img_size = 512
    classes = 11014
    scale = 30
    margin = 0.5
    fc_dim = 512
    epochs = 15
    batch_size = 32
    num_workers = 8
    model_name = 'tf_efficientnet_b4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * batch_size,  # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }


def run_training():
    df = pd.read_csv(TRAIN_CSV)

    labelencoder = LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    trainset = ShopeeDataset(df,
                             DATA_DIR,
                             transform=get_train_transforms(img_size=CFG.img_size))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    model = ShopeeModel()
    model.to(CFG.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG.scheduler_params['lr_start'])
    scheduler = ShopeeScheduler(optimizer, **CFG.scheduler_params)

    for epoch in range(CFG.epochs):
        avg_loss_train = engine.train_fn(model, trainloader, optimizer, scheduler, epoch, CFG.device)
        torch.save(model.state_dict(), MODEL_PATH + 'arcface_512x512_{}.pt'.format(CFG.model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
            MODEL_PATH + 'arcface_512x512_{}_checkpoints.pt'.format(CFG.model_name)
        )


if __name__ == "__main__":
    run_training()