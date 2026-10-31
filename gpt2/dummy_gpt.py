import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyGPTModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        #vocab size is the number of words in vocab, # embed_dim is the dimension of embedding 
        self.tok_embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])  
        #context_size is maximum number of tokens as inputs
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg)
                            for _ in range(cfg["n_blocks"])
                            ])
        self.final_norm = DummyLayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):

        batch_size, seq_length = in_idx.shape
        #shape is property (size is function)
        tok_embeds = self.tok_embedding(in_idx) 
        #we expect tok_embeddings to pull embeddings corresponding to all idx in in_idx
        pos_embeds = self.pos_embedding(torch.arange(seq_length, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):

    def __init__(self, embed_dim): 
        super().__init__()

    def forward(self, x):
        return x

