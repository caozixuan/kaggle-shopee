import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from .config import TITLE_PRE_TRAIN_CONFIG
from .utils import set_seed, get_device, JS_loss

from data.dataset import ShopeeTitleDataset
from model.pretrain_model.title_encoder import TitleEncoder


def pretrain():
    seed = TITLE_PRE_TRAIN_CONFIG['seed']
    cuda_enable = TITLE_PRE_TRAIN_CONFIG['cuda_enable']
    epochs = TITLE_PRE_TRAIN_CONFIG['epochs']
    batch_size = TITLE_PRE_TRAIN_CONFIG['batch_size']
    lr = TITLE_PRE_TRAIN_CONFIG['lr']
    weight_decay = TITLE_PRE_TRAIN_CONFIG['weight_decay']
    lr_scheduler_step = TITLE_PRE_TRAIN_CONFIG['lr_scheduler_step']

    set_seed(seed, cuda_enable)
    device = get_device(cuda_enable)

    print(device)
    dataset = ShopeeTitleDataset()
    dataset.set_device(device)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    in_dim = dataset.get_embed_size()
    model = TitleEncoder(in_dim).double()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("begins to train title encoder...")
    def train():
        for batch, (title, pos_title_list, neg_title_list) in enumerate(dataloader):
            title_c = model(title)
            pos_title_c_list = [model(item) for item in pos_title_list]
            neg_title_c_list = [model(item) for item in neg_title_list]
            loss = JS_loss(title_c, pos_title_c_list=pos_title_c_list, neg_title_c_list=neg_title_c_list)
            loss.backward()
            opt.step()
            print(f'batch: {batch} loss: {loss.item()}')

    for epoch in range(epochs):
        train()

    torch.save(model, 'title_encoder.pt')
    print("model is saved!")
