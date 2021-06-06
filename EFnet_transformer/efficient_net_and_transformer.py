import sys
sys.path.append('../input/utils-shopee')
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import timm

import engine
from dataset import ShopeeDataset
from custom_scheduler import ShopeeScheduler
from augmentations import get_train_transforms


from tqdm import tqdm
import os
import cv2

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