import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from .config import PRE_TRAIN_CONFIG
from .utils import set_seed, get_device, JS_loss

from data.dataset import MatchDataset
from model.pretrain_model.encoder import Encoder


def pretrain():
    seed = PRE_TRAIN_CONFIG['seed']
    cuda_enable = PRE_TRAIN_CONFIG['cuda_enable']
    epochs = PRE_TRAIN_CONFIG['epochs']
    batch_size = PRE_TRAIN_CONFIG['batch_size']
    lr = PRE_TRAIN_CONFIG['lr']
    weight_decay = PRE_TRAIN_CONFIG['weight_decay']
    lr_scheduler_step = PRE_TRAIN_CONFIG['lr_scheduler_step']

    set_seed(seed, cuda_enable)
    device = get_device(cuda_enable)

    print(device)
    dataset = MatchDataset()
    dataset.set_device(device)
    in_dim = dataset.get_title_embed_size()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    model = Encoder(in_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("begins to train encoder...")

    def train():
        for batch, (title, image, pos_title_list, pos_image_list, neg_title_list, neg_image_list) in enumerate(
                dataloader):
            c = model(title, image)
            pos_c_list = [model(pos_title_list[i], pos_image_list[i]) for i in range(len(pos_title_list))]
            neg_c_list = [model(neg_title_list[i], neg_image_list[i]) for i in range(len(neg_title_list))]
            loss = JS_loss(c, pos_c_list=pos_c_list, neg_c_list=neg_c_list)
            loss.backward()
            opt.step()
            print(f'batch: {batch} loss: {loss.item()}')

    for epoch in range(epochs):
        train()

    torch.save(model, 'encoder.pt')
    print("model is saved!")
