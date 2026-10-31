import logging
from dataclasses import dataclass

import torch
from tiktoken_tokenizer import create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings")
logger.setLevel(logging.INFO)

@dataclass
class EmbeddingConfig:
    vocab_size: int = 50257
    emb_size: int = 256
    max_length: int = 8
    batch_size: int = 4
    stride: int = 1
    shuffle: bool = False
    context_length: int = 8 

config = EmbeddingConfig()

dataloader = create_dataloader(batch_size=config.batch_size, max_length=config.max_length, stride=config.stride)
dataiter = iter(dataloader)
batch = next(dataiter)

token_embedding_layer = torch.nn.Embedding(config.vocab_size, config.emb_size)
token_embedding = token_embedding_layer(batch[0])
logger.info(token_embedding.shape)

pos_embedding_layer = torch.nn.Embedding(config.context_length, config.emb_size)
pos_embedding = pos_embedding_layer(torch.arange(config.context_length))
logger.info(pos_embedding.shape)

pos_embedding = pos_embedding.unsqueeze(0)

#pos_embedding = pos_embedding.expand(config.batch_size, -1, -1)
#or 
pos_embedding = pos_embedding.repeat(config.batch_size, 1, 1)
input_embeddings = token_embedding + pos_embedding
logger.info(input_embeddings.shape)