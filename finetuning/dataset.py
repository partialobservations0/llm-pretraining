import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):

    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):

        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        self.encoded_texts = [encoded_text[:max_length] for encoded_text in self.encoded_texts]
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
                              for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)   
    
    def _longest_encoded_length(self):
        max_length = 0
        for text in self.encoded_texts:
            if max_length < len(text):
                max_length = len(text)
        return max_length

train_dataset = SpamDataset(csv_file='./notebooks/train.csv', tokenizer = tiktoken.get_encoding('gpt2'))
print(len(train_dataset))