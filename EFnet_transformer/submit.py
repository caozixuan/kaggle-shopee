import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')

import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

import gc
import matplotlib.pyplot as plt
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors

class CFG:
    seed = 54
    classes = 11014
    scale = 30
    margin = 0.5
    model_name =  'tf_efficientnet_b4'
    fc_dim = 512
    img_size = 512
    batch_size = 20
    num_workers = 4
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '../input/utils-shopee/arcface_512x512_tf_efficientnet_b4_LR.pt'

def read_dataset():

    df = pd.read_csv('../input/shopee-product-matching/test.csv')
    df_cu = cudf.DataFrame(df)
    image_paths = '../input/shopee-product-matching/test_images/' + df['image']

    return df, df_cu, image_paths


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.seed)


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x) )

