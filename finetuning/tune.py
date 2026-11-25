
import tiktoken
import torch
from dataset import SpamDataset
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
torch.manual_seed(123)

loc = './notebooks/'
train_file = loc + 'train.csv'
test_file = loc + 'test.csv'
validation_file = loc + 'validation.csv'
tokenizer = tiktoken.get_encoding('gpt2')

train_dataset = SpamDataset(csv_file=train_file, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file=test_file, tokenizer=tokenizer)
validation_dataset = SpamDataset(csv_file=validation_file, tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, shuffle=True,
                          num_workers=0, batch_size=8)
test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=8)
validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=0, batch_size=8)

print(f"len training batches {len(train_loader)}")