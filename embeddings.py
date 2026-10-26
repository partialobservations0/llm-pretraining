import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger("embeddings")
logger.setLevel(logging.INFO)

import torch
from tiktoken_tokenizer import TokenizerV2, TokenDataset, create_dataloader

vocab_size = TokenizerV2().vocab_size
emb_size = 2
max_length = 4
batch_size = 1
stride = 1
embedding = torch.nn.Embedding(vocab_size, emb_size)

dataloader = create_dataloader(batch_size=batch_size, max_length=max_length, stride=stride)
dataiter = iter(dataloader)
batch = next(dataiter)
embedded = embedding(batch[0])


