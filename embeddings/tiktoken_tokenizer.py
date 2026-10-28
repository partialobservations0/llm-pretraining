import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tiktoken")
logger.setLevel(logging.INFO)
from importlib.metadata import version 
from words import read_file
import tiktoken
logger.info("tiktoken version: " + version("tiktoken"))
from Tokenizer import Tokenizer

import torch
from torch.utils.data import DataLoader, Dataset

text = read_file()
tokenizer = tiktoken.get_encoding("gpt2")

def test_tiktoken(tokenizer, text):
    tokens = tokenizer.encode(text[:10])
    logger.info(tokens)
    string = tokenizer.decode(tokens)
    logger.info(string)

def create_dataloader(batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0):

    dataset = TokenDataset(TokenizerV2(), read_file(), max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                          drop_last=drop_last, num_workers=num_workers)
    return dataloader    

class TokenizerV2:
    def __init__(self, **kwargs):
        if "tokenizer" not in kwargs:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
        else:
            self.tokenizer = kwargs["tokenizer"]
            self.vocab_size = len(self.tokenizer.vocab)if self.tokenizer.vocab else 0
        logger.info("Tokenizer initialized")

        if "text" in kwargs:
            self.text = kwargs["text"]
        else:
            self.text = read_file()
        logger.info("Text loaded")

    def encode(self, text):
        if not text:
            text = self.text
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class TokenDataset(Dataset):

    def __init__(self, tokenizer, text, max_length, stride=1):
        self.text = text
        self.tokens = tokenizer.encode(text)
        self.input_ids = []
        self.target_ids = []
        self.max_length = max_length
        self.stride = stride
        self.build_pairs()

    def build_pairs(self):
        for i in range(0, len(self.tokens) - self.max_length, self.stride):
            self.input_ids.append(torch.tensor(self.tokens[i:i+self.max_length]))
            self.target_ids.append(torch.tensor(self.tokens[i+1:i+self.max_length+1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



if __name__ == "__main__":
    tokenizer = Tokenizer(filepath='the-verdict.txt')
    t = TokenizerV2(tokenizer=tokenizer)
    t = TokenizerV2()
    #print(t.encode())
    #print(t.decode(t.encode()))
    dataloader = create_dataloader(batch_size=1,max_length=4,stride=4)
    data_iter = iter(dataloader)
    batch = next(data_iter)

    print(batch)
    second = next(data_iter)
    print(second)