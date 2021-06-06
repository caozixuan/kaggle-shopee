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


class CONFIG:
    seed = 54
    img_size = 512
    batch_size = 4
    num_workers = 1
    n_classes = 11014
    scale = 30
    margin = 0.5
    fc_dim = 256
    epochs = 15
    pretrain = True
    use_fc = True
    dropout = 0.1
    img_encoder_name = 'tf_efficientnet_b4'
    text_encoder_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(text_encoder_name)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model_name = 'efficient_net_transformer'
    model_path = '...'
    scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * batch_size,  # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }


CHECK_SUB = False
GET_CV = True

test = pd.read_csv('../input/shopee-product-matching/test.csv')
if len(test)>3:
    GET_CV = False
else:
    print('this submission notebook will compute CV score, but commit notebook will not')


def read_dataset():
    if GET_CV:
        df = pd.read_csv('../input/shopee-product-matching/train.csv')
        tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
        df['matches'] = df['label_group'].map(tmp)
        df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
        if CHECK_SUB:
            df = pd.concat([df, df], axis=0)
            df.reset_index(drop=True, inplace=True)
        root_dir = '../input/shopee-product-matching/train_images/'
        df_cu = cudf.DataFrame(df)
    else:
        df = pd.read_csv('../input/shopee-product-matching/test.csv')
        df_cu = cudf.DataFrame(df)
        root_dir = '../input/shopee-product-matching/test_images/'

    return df, df_cu, root_dir


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CONFIG.seed)


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


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0, device='cuda'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.device = device

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output


class ShopDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # label = row.label_group
        text = row.title

        text = CONFIG.TOKENIZER(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        img_path = os.path.join(self.root_dir, row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return (input_ids, attention_mask, image, torch.tensor(1).long())


class ShopeeModel(nn.Module):
    def __init__(self, n_classes, img_encoder_name, text_encoder_name, fc_dim, margin, scale, pretrain=True, use_fc=True, dropout=0.1, device='cpu'):
        super(ShopeeModel, self).__init__()
        self.n_classes = n_classes
        self.img_encoder_name = img_encoder_name
        self.text_encoder_name = text_encoder_name
        self.fc_dim = fc_dim
        self.margin = margin
        self.scale = scale
        self.use_fc = use_fc
        self.device = device

        print('Building Image Encoder Backbone for {} model'.format(img_encoder_name))

        self.img_backbone = timm.create_model(img_encoder_name, pretrained=pretrain)
        img_in_features = self.img_backbone.classifier.in_features
        self.img_backbone.classifier = nn.Identity()
        self.img_backbone.global_pool = nn.Identity()
        self.img_pooling = nn.AdaptiveAvgPool2d(1)

        self.transformer = transformers.AutoModel.from_pretrained(self.text_encoder_name)
        text_in_features = self.transformer.config.hidden_size

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=dropout)

            self.text_fc = nn.Linear(text_in_features, fc_dim)
            self.img_fc = nn.Linear(img_in_features, fc_dim)

            self.text_bn = nn.BatchNorm1d(fc_dim)
            self.img_bn = nn.BatchNorm1d(fc_dim)

            self.relu = nn.ReLU()
            self._init_params()

            text_in_features = fc_dim
            img_in_features = fc_dim

        in_features = img_in_features + text_in_features

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0,
            device=self.device
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.text_fc.weight)
        nn.init.constant_(self.text_fc.bias, 0)
        nn.init.constant_(self.text_bn.weight, 1)
        nn.init.constant_(self.text_bn.bias, 0)
        nn.init.xavier_normal_(self.img_fc.weight)
        nn.init.constant_(self.img_fc.bias, 0)
        nn.init.constant_(self.img_bn.weight, 1)
        nn.init.constant_(self.img_bn.bias, 0)

    def extract_img_features(self, x):
        batch_size = x.shape[0]
        x = self.img_backbone(x)
        x = self.img_pooling(x).view(batch_size, -1)

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.img_fc(x)
            x = self.img_bn(x)
        return x

    def extract_text_features(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        features = x[0]
        features = features[:, 0, :]

        if self.use_fc:
            features = self.dropout(features)
            features = self.text_fc(features)
            features = self.text_bn(features)
            features = self.relu(features)
        return features

    def forward(self, input_ids, attention_mask, image, label):
        img_features = self.extract_img_features(image)
        text_feature = self.extract_text_features(input_ids, attention_mask)
        features = torch.cat([img_features, text_feature], dim=-1)
        if self.training:
            logits = self.final(features, label)
            return logits
        else:
            return features


def get_neighbors(df, embeddings, KNN=50):
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    threshold = 4.5
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)

    del model, distances, indices
    gc.collect()
    return df, predictions


def get_neighbors_knn(df, embeddings, KNN=50):
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    if GET_CV:
        thresholds = list(np.arange(0, 2, 0.1))

        scores = []
        for threshold in thresholds:
            predictions = []
            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k, idx]
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)
            df['pred_matches'] = predictions
            df['f1'] = f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            print(f'Our f1 score for threshold {threshold} is {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')

        # Use threshold
        predictions = []
        for k in range(embeddings.shape[0]):
            # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
            idx = np.where(distances[k,] < 0.60)[0]
            ids = indices[k, idx]
            posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
            predictions.append(posting_ids)

    # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
    else:
        predictions = []
        for k in tqdm(range(embeddings.shape[0])):
            idx = np.where(distances[k,] < 0.60)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    del model, distances, indices
    gc.collect()
    return df, predictions


def get_test_transforms():
    return albumentations.Compose([
        albumentations.Resize(CONFIG.img_size, CONFIG.img_size, always_apply=True),
        albumentations.Normalize(),
        ToTensorV2(p=1.0)
    ])


def get_embeddings(data, root_dir):

    model = ShopeeModel(n_classes=CONFIG.n_classes, img_encoder_name=CONFIG.img_encoder_name, text_encoder_name=CONFIG.text_encoder_name, fc_dim=CONFIG.fc_dim, margin=CONFIG.margin, scale=CONFIG.scale, pretrain=False).to(CONFIG.device)
    model.load_state_dict(torch.load(CONFIG.model_path))
    model.eval()

    dataset = ShopDataset(data, root_dir, transform=get_test_transforms())  # (image_paths=image_paths, transforms=get_test_transforms())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers
    )

    embeds = []
    with torch.no_grad():
        for input_ids, attention_mask, image, label in tqdm(dataloader):
            input_ids = input_ids.to(CONFIG.device)
            attention_mask = attention_mask.to(CONFIG.device)
            image = image.to(CONFIG.device)
            label = label.to(CONFIG.device)
            features = model(input_ids, attention_mask, image, label)
            embeddings = features.detach().cpu().numpy()
            embeds.append(embeddings)

    del model
    embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {embeddings.shape}')
    del embeds
    gc.collect()
    return embeddings


df, df_cu, root_dir = read_dataset()
df.head()

embeddings = get_embeddings(df, root_dir)

df, predictions = get_neighbors_knn(df, embeddings)

if not GET_CV:
    train_predictions = [' '.join(preds) for preds in predictions]
    df['matches'] = train_predictions
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
else:
    df['matches'] = predictions
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)