# efficient net parts

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

# import engine
# from dataset import ShopeeDataset
# from custom_scheduler import ShopeeScheduler
# from augmentations import get_train_transforms


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


class CFG:
    seed = 54
    classes = 11014
    scale = 30
    margin = 0.5
    model_name = 'tf_efficientnet_b4'
    fc_dim = 512
    img_size = 512
    batch_size = 20
    num_workers = 4
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '../input/utils-shopee/arcface_512x512_tf_efficientnet_b4_LR.pt'


class CONFIG:
    seed = 54
    img_size = 512
    batch_size = 16
    num_workers = 1
    n_classes = 11014
    scale = 30
    margin = 0.5
    fc_dim = 512
    epochs = 15
    pretrain = True
    use_fc = True
    dropout = 0.1

    img_encoder_name = 'tf_efficientnet_b4'
    img_encoder_path = '../input/utils-shopee/arcface_512x512_tf_efficientnet_b4_LR.pt'

    text_encoder_name = '../input/sentence-transformer-models/paraphrase-xlm-r-multilingual-v1/0_Transformer'
    text_encoder_path = '../input/best-multilingual-model/sentence_transfomer_xlm_best_loss_num_epochs_25_arcface.bin'

    TOKENIZER = transformers.AutoTokenizer.from_pretrained(text_encoder_name)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model_name = 'efficient_net_transformer'

    model_params = {
        'n_classes': 11014,
        'model_name': text_encoder_name,
        'use_fc': False,
        'fc_dim': 512,
        'dropout': 0.3,
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
    return ' '.join( np.unique(x))


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

        text = CONFIG.TOKENIZER(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        img_path = os.path.join(self.root_dir, row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return (input_ids, attention_mask, image, torch.tensor(1).long())


class TEXT_Model(nn.Module):
    def __init__(self,
                 n_classes = CONFIG.n_classes,
                 model_name='bert-base-uncased',
                 use_fc=False,
                 fc_dim=CONFIG.fc_dim,
                 dropout=0.0):
        super(TEXT_Model, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        final_in_features = self.transformer.config.hidden_size

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids, attention_mask):
        feature = self.extract_feat(input_ids, attention_mask)
        return F.normalize(feature)

    def extract_feat(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        features = x[0]
        features = features[:, 0, :]

        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)

        return features


class IMG_Model(nn.Module):
    def __init__(
            self,
            n_classes=CONFIG.n_classes,
            model_name=CONFIG.img_encoder_name,
            fc_dim=CONFIG.fc_dim,
            margin=CONFIG.margin,
            scale=CONFIG.scale,
            use_fc=True,
            pretrained=True):

        super(IMG_Model, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        features = self.extract_features(image)
        if self.training:
            logits = self.final(features, label)
            return logits
        else:
            return features

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x


def get_image_neighbors(df, embeddings, KNN=50):
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


def get_test_transforms():
    return albumentations.Compose([
        albumentations.Resize(CFG.img_size, CFG.img_size, always_apply=True),
        albumentations.Normalize(),
        ToTensorV2(p=1.0)
    ])


def get_neighbours_cos_sim(df, embeddings):
    '''
    When using cos_sim use normalized features else use normal features
    '''
    embeddings = cupy.array(embeddings)

    if GET_CV:
        thresholds = list(np.arange(0.5, 0.7, 0.05))

        scores = []
        for threshold in thresholds:

            ################################################# Code for Getting Preds #########################################
            preds = []
            CHUNK = 1024 * 4

            print('Finding similar titles...for threshold :', threshold)
            CTS = len(embeddings) // CHUNK
            if len(embeddings) % CHUNK != 0: CTS += 1

            for j in range(CTS):
                a = j * CHUNK
                b = (j + 1) * CHUNK
                b = min(b, len(embeddings))

                cts = cupy.matmul(embeddings, embeddings[a:b].T).T

                for k in range(b - a):
                    IDX = cupy.where(cts[k,] > threshold)[0]
                    o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
                    o = ' '.join(o)
                    preds.append(o)
            ######################################################################################################################
            df['pred_matches'] = preds
            df['f1'] = f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            print(f'Our f1 score for threshold {threshold} is {score}')
            scores.append(score)

        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')

    else:
        preds = []
        CHUNK = 1024 * 4
        threshold = 0.7

        print('Finding similar texts...for threshold :', threshold)
        CTS = len(embeddings) // CHUNK
        if len(embeddings) % CHUNK != 0: CTS += 1

        for j in range(CTS):
            a = j * CHUNK
            b = (j + 1) * CHUNK
            b = min(b, len(embeddings))
            print('chunk', a, 'to', b)

            cts = cupy.matmul(embeddings, embeddings[a:b].T).T

            for k in range(b - a):
                IDX = cupy.where(cts[k,] > threshold)[0]
                o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
                preds.append(o)

    return df, preds


def get_embeddings(data, root_dir):

    img_model = IMG_Model(pretrained=False).to(CONFIG.device)
    img_model.load_state_dict(torch.load(CONFIG.img_encoder_path))
    img_model.eval()
    img_model = img_model.to(CONFIG.device)

    text_model = TEXT_Model(**CONFIG.model_params)
    text_model.eval()
    text_model.load_state_dict(dict(list(torch.load(CONFIG.text_encoder_path).items())[:-1]))
    text_model = text_model.to(CONFIG.device)

    dataset = ShopDataset(data, root_dir, transform=get_test_transforms())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers
    )

    img_embeds = []
    text_embeds = []
    with torch.no_grad():
        for input_ids, attention_mask, image, label in tqdm(dataloader):
            image = image.to(CONFIG.device)
            label = label.to(CONFIG.device)
            img_features = img_model(image, label)
            text_features = text_model(input_ids, attention_mask)
            image_embeddings = img_features.detach().cpu().numpy()
            text_embeddings = text_features.detach().cpu().numpy()
            img_embeds.append(image_embeddings)
            text_embeds.append(text_embeddings)

    del img_model
    del text_model
    image_embeddings = np.concatenate(img_embeds)
    text_embeddings = np.concatenate(text_embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del img_embeds
    del text_embeds
    gc.collect()
    return image_embeddings, text_embeddings


# def get_text_predictions(df, max_features=25_000):
#     model = TfidfVectorizer(stop_words='english',
#                             binary=True,
#                             max_features=max_features)
#     text_embeddings = model.fit_transform(df_cu['title']).toarray()
#
#     print('Finding similar titles...')
#     CHUNK = 1024 * 4
#     CTS = len(df) // CHUNK
#     if (len(df) % CHUNK) != 0:
#         CTS += 1
#
#     preds = []
#     for j in range(CTS):
#         a = j * CHUNK
#         b = (j + 1) * CHUNK
#         b = min(b, len(df))
#         print('chunk', a, 'to', b)
#
#         # COSINE SIMILARITY DISTANCE
#         cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T
#         for k in range(b - a):
#             IDX = cupy.where(cts[k,] > 0.75)[0]
#             o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
#             preds.append(o)
#
#     del model, text_embeddings
#     gc.collect()
#     return preds


df, df_cu, root_dir = read_dataset()
df.head()


image_embeddings, text_embeddings = get_embeddings(df, root_dir)
df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=50 if len(df)>3 else 3)
df, text_predictions = get_neighbours_cos_sim(df,text_embeddings)


df['image_predictions'] = image_predictions
df['text_predictions'] = text_predictions
df['matches'] = df.apply(combine_predictions, axis=1)
df[['posting_id', 'matches']].to_csv('submission.csv', index=False)

