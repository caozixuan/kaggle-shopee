import torch.nn as nn
import torch.nn.functional as F

from .config import IMAGE_ENCODER_CONFIG
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.freeze_resnet()
        fc_inputs = self.resnet.fc.in_features
        hidden_dim = IMAGE_ENCODER_CONFIG['hidden_dim']
        out_dim = IMAGE_ENCODER_CONFIG['out_dim']
        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_inputs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feature):
        c = self.resnet(feature)
        return F.normalize(c, dim=1)

    def freeze_resnet(self):
        for key, items in self.resnet.named_parameters():
            items.requires_grad = False

