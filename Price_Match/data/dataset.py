import os
import torch
import numpy as np
import pandas as pd

from .config import DATA_CONFIG, IMAGE_DATA_CONFIG, TITLE_DATA_CONFIG
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA


class ShopeeImageDataset(Dataset):
    def __init__(self):
        super(ShopeeImageDataset, self).__init__()
        image_file_path = DATA_CONFIG['image_path']
        file_path = DATA_CONFIG['match_path']
        description_set_name = DATA_CONFIG['train_match_set_name']
        image_set_name = DATA_CONFIG['train_images_filename']
        d_file = os.path.join(file_path, description_set_name)
        i_file = os.path.join(image_file_path, image_set_name)
        if not (os.path.exists(d_file) and os.path.exists(i_file)):
            print(f'no such file or directory {d_file} or {i_file}')
            exit(0)
        self.df = pd.read_csv(d_file)
        ndata = len(self.df)
        nlist = range(0, ndata)
        self.df['id'] = nlist
        self.img_path = i_file + '/' + self.df['image'].values
        self.group = self.df.groupby('label_group')
        self.device = torch.device("cpu")
        self.negative_sample_size = IMAGE_DATA_CONFIG['negative_sample_size']
        self.positive_sample_size = IMAGE_DATA_CONFIG['positive_sample_size']
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomCrop(size=(256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def set_device(self, device):
        assert device == torch.device("cpu") or device == torch.device("cuda"), f"device error: {device}"
        self.device = device

    def __getitem__(self, index):
        group_id = self.df['label_group'].iloc[index]
        img_group = self.group.get_group(group_id)
        pos_id = np.random.choice(img_group['id'].values, self.positive_sample_size)
        neg_id = np.random.choice(self.df['id'].values, self.negative_sample_size)
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img).to(self.device)
        pos_img_list = []
        neg_img_list = []
        for id in pos_id:
            pos_img = Image.open(self.img_path[id]).convert('RGB')
            pos_img = self.transform(pos_img).to(self.device)
            pos_img_list.append(pos_img)
        for id in neg_id:
            neg_img = Image.open(self.img_path[id]).convert('RGB')
            neg_img = self.transform(neg_img).to(self.device)
            neg_img_list.append(neg_img)
        return (img, pos_img_list, neg_img_list)

    def __len__(self):
        return len(self.img_path)


class ShopeeTitleDataset(Dataset):
    def __init__(self):
        super(ShopeeTitleDataset, self).__init__()
        file_path = DATA_CONFIG['match_path']
        description_set_name = DATA_CONFIG['train_match_set_name']
        d_file = os.path.join(file_path, description_set_name)
        if not os.path.exists(d_file):
            print(f'no such file or directory {d_file}')
            exit(0)
        self.df = pd.read_csv(d_file)
        ndata = len(self.df)
        nlist = range(0, ndata)
        self.device = torch.device("cpu")
        self.df['id'] = nlist
        self.group = self.df.groupby('label_group')
        self.embed_size = TITLE_DATA_CONFIG['embed_size']
        self.negative_sample_size = TITLE_DATA_CONFIG['negative_sample_size']
        self.positive_sample_size = TITLE_DATA_CONFIG['positive_sample_size']

        embed_save_path = TITLE_DATA_CONFIG['embed_save_path']
        if not os.path.exists(embed_save_path):
            self.title_embed = torch.tensor(self.tfidf_pca(self.embed_size))
            torch.save(self.title_embed, embed_save_path)
        else:
            self.title_embed = torch.load(embed_save_path)

    def get_embed_size(self):
        return self.embed_size

    def set_device(self, device):
        assert device == torch.device("cpu") or device == torch.device("cuda"), f"device error: {device}"
        self.device = device

    def tfidf_pca(self, embed_size):
        corpus = self.df['title'].values
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        print("using pca")
        pca = PCA(n_components=embed_size, random_state=0)
        tfidf_pca = pca.fit_transform(tfidf.toarray())
        print("pca end")
        return tfidf_pca

    def __getitem__(self, index):
        group_id = self.df['label_group'].iloc[index]
        title_group = self.group.get_group(group_id)
        pos_id = np.random.choice(title_group['id'].values, self.positive_sample_size)
        neg_id = np.random.choice(self.df['id'].values, self.negative_sample_size)

        title = self.title_embed[index].to(self.device)
        pos_title_list = []
        neg_title_list = []
        for id in pos_id:
            pos_title_list.append(self.title_embed[id].to(self.device))
        for id in neg_id:
            neg_title_list.append(self.title_embed[id].to(self.device))
        return (title, pos_title_list, neg_title_list)

    def __len__(self):
        return len(self.df)


class MatchDataset(Dataset):
    def __init__(self):
        super(MatchDataset, self).__init__()
        image_file_path = DATA_CONFIG['image_path']
        file_path = DATA_CONFIG['match_path']
        description_set_name = DATA_CONFIG['train_match_set_name']
        image_set_name = DATA_CONFIG['train_images_filename']
        d_file = os.path.join(file_path, description_set_name)
        i_file = os.path.join(image_file_path, image_set_name)
        if not (os.path.exists(d_file) and os.path.exists(i_file)):
            print(f'no such file or directory {d_file} or {i_file}')
            exit(0)
        self.df = pd.read_csv(d_file)
        ndata = len(self.df)
        nlist = range(0, ndata)
        self.device = torch.device("cpu")
        self.df['id'] = nlist
        self.img_path = i_file + '/' + self.df['image'].values
        self.group = self.df.groupby('label_group')
        self.embed_size = TITLE_DATA_CONFIG['embed_size']
        self.negative_sample_size = TITLE_DATA_CONFIG['negative_sample_size']
        self.positive_sample_size = TITLE_DATA_CONFIG['positive_sample_size']
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        embed_save_path = TITLE_DATA_CONFIG['embed_save_path']
        if not os.path.exists(embed_save_path):
            self.title_embed = torch.tensor(self.tfidf_pca(self.embed_size))
            torch.save(self.title_embed, embed_save_path)
        else:
            self.title_embed = torch.load(embed_save_path)

    def get_title_embed_size(self):
        return self.embed_size

    def set_device(self, device):
        assert device == torch.device("cpu") or device == torch.device("cuda"), f"device error: {device}"
        self.device = device

    def tfidf_pca(self, embed_size):
        corpus = self.df['title'].values
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        pca = PCA(n_components=embed_size, random_state=0)
        tfidf_pca = pca.fit_transform(tfidf.toarray())
        return tfidf_pca

    def __getitem__(self, index):
        group_id = self.df['label_group'].iloc[index]
        group = self.group.get_group(group_id)
        pos_id = np.random.choice(group['id'].values, self.positive_sample_size)
        neg_id = np.random.choice(self.df['id'].values, self.negative_sample_size)

        title = self.title_embed[index].to(self.device)
        image = Image.open(self.img_path[index]).convert('RGB')
        image = self.transform(image).to(self.device)
        pos_title_list = []
        pos_image_list = []
        neg_title_list = []
        neg_image_list = []
        for id in pos_id:
            pos_title_list.append(self.title_embed[id].to(self.device))
            pos_img = Image.open(self.img_path[id]).convert('RGB')
            pos_img = self.transform(pos_img).to(self.device)
            pos_image_list.append(pos_img)
        for id in neg_id:
            neg_title_list.append(self.title_embed[id].to(self.device))
            neg_img = Image.open(self.img_path[id]).convert('RGB')
            neg_img = self.transform(neg_img).to(self.device)
            neg_image_list.append(neg_img)
        return (title, image, pos_title_list, pos_image_list, neg_title_list, neg_image_list)

    def __len__(self):
        return len(self.df)


class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        image_file_path = DATA_CONFIG['image_path']
        file_path = DATA_CONFIG['match_path']
        description_set_name = DATA_CONFIG['test_match_set_name']
        image_set_name = DATA_CONFIG['test_images_filename']
        d_file = os.path.join(file_path, description_set_name)
        i_file = os.path.join(image_file_path, image_set_name)
        if not (os.path.exists(d_file) and os.path.exists(i_file)):
            print(f'no such file or directory {d_file} or {i_file}')
            exit(0)
        self.df = pd.read_csv(d_file)
        self.device = torch.device("cpu")
        self.img_path = i_file + '/' + self.df['image'].values
        self.embed_size = TITLE_DATA_CONFIG['embed_size']
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        embed_save_path = TITLE_DATA_CONFIG['embed_save_path']
        if not os.path.exists(embed_save_path):
            self.title_embed = torch.tensor(self.tfidf_pca(self.embed_size))
            torch.save(self.title_embed, embed_save_path)
        else:
            self.title_embed = torch.load(embed_save_path)

    def get_title_embed_size(self):
        return self.embed_size

    def set_device(self, device):
        assert device == torch.device("cpu") or device == torch.device("cuda"), f"device error: {device}"
        self.device = device

    def tfidf_pca(self, embed_size):
        corpus = self.df['title'].values
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        pca = PCA(n_components=embed_size, random_state=0)
        tfidf_pca = pca.fit_transform(tfidf.toarray())
        return tfidf_pca

    def get_posting_id(self, index):
        return self.df['posting_id'].iloc[index]

    def posting_id(self):
        return self.df['posting_id']

    def __getitem__(self, index):
        title = self.title_embed[index].to(self.device)
        image = Image.open(self.img_path[index]).convert('RGB')
        image = self.transform(image).to(self.device)
        return title, image

    def __len__(self):
        return len(self.df)


class OldMatchDataset(Dataset):
    def __init__(self):
        super(OldMatchDataset, self).__init__()
        file_path = DATA_CONFIG['match_path']
        description_set_name = DATA_CONFIG['train_match_set_name']
        image_set_name = DATA_CONFIG['train_images_filename']
        d_file = os.path.join(file_path, description_set_name)
        i_file = os.path.join(file_path, image_set_name)
        if not (os.path.exists(d_file) and os.path.exists(i_file)):
            print(f'no such file or directory {d_file} or {i_file}')
            exit(0)
        self.df = pd.read_csv(d_file, header=1)
        img_path = self.df['image'].values
        self.img_path = i_file + '/' + img_path
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        posting_id = self.df['posting_id'].iloc[item]
        description = self.df['title'].iloc[item]
        label_group = self.df['label_group'].iloc[item]
        img = Image.open(self.img_path[item]).convert('RGB')
        img = self.transform(img)
        return (posting_id, description, img, label_group)


class DescriptionDataset(Dataset):
    def __init__(self):
        super(DescriptionDataset, self).__init__()
        file_path = DATA_CONFIG['match_path']
        set_name = DATA_CONFIG['train_match_set_name']
        file = os.path.join(file_path, set_name)
        if not os.path.exists(file):
            print(f'no such file or directory {file}')
            exit(0)
        self.df = pd.read_csv(file, header=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        posting_id = self.df['posting_id'].iloc[item]
        description = self.df['title'].iloc[item]
        label_group = self.df['label_group'].iloc[item]
        return (posting_id, description, label_group)


def load_test_images():
    file_path = DATA_CONFIG['image_path']
    set_name = DATA_CONFIG['test_images_filename']
    file = os.path.join(file_path, set_name)
    if not os.path.exists(file):
        print(f'no such file or directory {file}')
        exit(0)
    # load images


def load_descriptions():
    file_path = DATA_CONFIG['image_path']
    set_name = DATA_CONFIG['test_images_filename']
    file = os.path.join(file_path, set_name)
    if not os.path.exists(file):
        print(f'no such file or directory {file}')
        exit(0)
    # load descriptions
