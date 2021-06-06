import torch

model = torch.load('image_encoder.pt')
model.to(torch.device("cuda"))
model.eval()
print("success")