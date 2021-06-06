class ShopeeDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv.reset_index()

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title

        text = TOKENIZER(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        return input_ids, attention_mask, torch.tensor(row.label_group)