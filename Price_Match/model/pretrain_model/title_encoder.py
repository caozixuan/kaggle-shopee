import torch.nn as nn
import torch.nn.functional as F

from .config import TITLE_ENCODER_CONFIG


class TitleEncoder(nn.Module):
    def __init__(self, in_dim):
        super(TitleEncoder, self).__init__()
        hidden_dim = TITLE_ENCODER_CONFIG['hidden_dim']
        out_dim = TITLE_ENCODER_CONFIG['out_dim']
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, title):
        c = self.fc(title)
        return F.normalize(c)