import torch
import time
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from .config import IMAGE_PRE_TRAIN_CONFIG
from .utils import set_seed, get_device, JS_loss

from data.dataset import ShopeeImageDataset
from model.pretrain_model.image_encoder import ImageEncoder


def pretrain():
    seed = IMAGE_PRE_TRAIN_CONFIG['seed']
    cuda_enable = IMAGE_PRE_TRAIN_CONFIG['cuda_enable']
    epochs = IMAGE_PRE_TRAIN_CONFIG['epochs']
    batch_size = IMAGE_PRE_TRAIN_CONFIG['batch_size']
    lr = IMAGE_PRE_TRAIN_CONFIG['lr']
    weight_decay = IMAGE_PRE_TRAIN_CONFIG['weight_decay']
    lr_scheduler_step = IMAGE_PRE_TRAIN_CONFIG['lr_scheduler_step']

    set_seed(seed, cuda_enable)
    device = get_device(cuda_enable)
    print(device)

    dataset = ShopeeImageDataset()
    dataset.set_device(device)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    model = ImageEncoder()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("begins to train image encoder...")

    def train():
        start_t = time.time()
        for batch, (img, pos_img_list, neg_img_list) in enumerate(dataloader):
            img_c = model(img)
            pos_img_c_list = [model(item) for item in pos_img_list]
            neg_img_c_list = [model(item) for item in neg_img_list]
            loss = JS_loss(img_c, pos_img_c_list=pos_img_c_list, neg_img_c_list=neg_img_c_list)
            loss.backward()
            opt.step()
            if batch % 10 == 0:
                print(f'batch: {batch} loss: {loss.item()} time: {time.time()-start_t}')
                start_t = time.time()

    for epoch in range(epochs):
        train()

    torch.save(model, 'image_encoder.pt')
    print("model is saved!")