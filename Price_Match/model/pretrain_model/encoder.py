import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IMAGE_ENCODER_CONFIG, TITLE_ENCODER_CONFIG, ENCODER_CONFIG
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.freeze_resnet()
        fc_inputs = self.resnet.fc.in_features
        hidden_dim = IMAGE_ENCODER_CONFIG['hidden_dim']
        self.out_dim = IMAGE_ENCODER_CONFIG['out_dim']
        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_inputs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, self.out_dim)
        )

    def forward(self, feature):
        c = self.resnet(feature)
        return F.normalize(c, dim=1)

    def freeze_resnet(self):
        for key, items in self.resnet.named_parameters():
            items.requires_grad = False

    def get_out_dim(self):
        return self.out_dim


class TitleEncoder(nn.Module):
    def __init__(self, in_dim=128):
        super(TitleEncoder, self).__init__()
        hidden_dim = TITLE_ENCODER_CONFIG['hidden_dim']
        self.out_dim = TITLE_ENCODER_CONFIG['out_dim']
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )

    def forward(self, title):
        c = self.fc(title)
        return F.normalize(c, dim=1)

    def get_out_dim(self):
        return self.out_dim


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.title_encoder = TitleEncoder(in_dim).double()
        self.image_encoder = ImageEncoder()
        self.hidden_dim = self.title_encoder.get_out_dim() + self.image_encoder.get_out_dim()
        self.out_dim = ENCODER_CONFIG['out_dim']
        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, title, image):
        title_c = self.title_encoder(title)
        image_c = self.image_encoder(image)
        c = torch.cat([title_c, image_c], dim=1).float()
        c = self.fc(c)
        return F.normalize(c, dim=1)

    def get_out_dim(self):
        return self.out_dim
