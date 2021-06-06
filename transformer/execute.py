import math
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Visuals and CV2
import cv2

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import KFold, train_test_split

#torch
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, lr_scheduler

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from dataset import ShopeeDataset
from arcmargin import ArcMarginProduct
from model import ShopeeNet
from utils import AverageMeter

NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 32
EPOCHS = 5
SEED = 2020
LR = 5e-5

device = torch.device('cuda')

################################################# MODEL ####################################################################

transformer_model = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
TOKENIZER = transformers.AutoTokenizer.from_pretrained(transformer_model)

################################################ Metric Loss and its params #######################################################
loss_module = 'arcface' #'softmax'
s = 30.0
m = 0.5
ls_eps = 0.0
easy_margin = False

############################################################################################################################
model_params = {
    'n_classes':11014,
    'model_name':transformer_model,
    'pooling':'clf',
    'use_fc':False,
    'fc_dim':512,
    'dropout':0.0,
    'loss_module':loss_module,
    's':30.0,
    'margin':0.50,
    'ls_eps':0.0,
    'theta_zero':0.785
}


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:

        batch_size = d[0].shape[0]

        input_ids = d[0]
        attention_mask = d[1]
        targets = d[2]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(input_ids, attention_mask, targets)

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

    return loss_score


data = pd.read_csv('../input/shopee-product-matching/train.csv')
data['filepath'] = data['image'].apply(lambda x: os.path.join('../input/shopee-product-matching/', 'train_images', x))

encoder = LabelEncoder()
data['label_group'] = encoder.fit_transform(data['label_group'])


def run():
    # Defining DataSet
    train_dataset = ShopeeDataset(
        csv=data
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=NUM_WORKERS
    )

    # Defining Device
    device = torch.device("cuda")

    # Defining Model for specific fold
    model = ShopeeNet(**model_params)
    model.to(device)

    # DEfining criterion
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_parameters, lr=LR)

    # Defining LR SCheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 2,
        num_training_steps=len(train_loader) * EPOCHS
    )

    # THE ENGINE LOOP
    best_loss = 10000
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)

        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            torch.save(model.state_dict(), f'sentence_transfomer_xlm_best_loss_num_epochs_{EPOCHS}_{loss_module}.bin')


if __name__ == "__main__":
    run()
