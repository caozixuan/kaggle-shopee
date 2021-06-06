import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import TestDataset

encoder = torch.load('encoder.pt')
out_dim = encoder.get_out_dim()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TestDataset()
dataset.set_device(device)
n_posting = len(dataset)

embeddings = torch.zeros(size=(n_posting, out_dim), dtype=torch.float)

print("begin embed")
for i, (title, image) in tqdm(enumerate(dataset)):
    title = torch.unsqueeze(title, dim=0)
    image = torch.unsqueeze(image, dim=0)
    embeddings[i] = encoder(title, image)

score_matrix = torch.mm(embeddings, embeddings.t())
values, indices = torch.topk(score_matrix, k=50, largest=True, sorted=True)

print("top 50!")
indices = np.array(indices).reshape(-1)
pred_posting_id = dataset.get_posting_id(indices)
pred_posting_id = np.array(pred_posting_id).reshape(-1, 50)

df_output = dataset.posting_id()
df_output['matches'] = pred_posting_id
df_output.to_csv('submission.csv')
