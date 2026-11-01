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
                            for _ in range(cfg["n_heads"])
                            ])
        self.final_norm = DummyLayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):

        _batch_size, seq_length = in_idx.shape
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
        # embed_dim is the dim of last dimension
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(embed_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):

        xmean = x.mean(dim=-1, keepdim=True)
        xvar = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased = False only to match GPT-2 specifications 
                                                            # it switches off Bessel's correction (divide by n-1 instead of n)
        norm_x = (x - xmean) / torch.sqrt(xvar + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class MyFeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class ExampleShortcutConnection(nn.Module):

    def __init__(self,layer_size, use_shortcut):

        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
                                    nn.Sequential(nn.Linear(layer_size[0], layer_size[1]), GELU()),
                                    nn.Sequential(nn.Linear(layer_size[1], layer_size[2]), GELU()),
                                    nn.Sequential(nn.Linear(layer_size[2], layer_size[3]), GELU()),
                                    nn.Sequential(nn.Linear(layer_size[3], layer_size[4]), GELU())
                                    ])

    def forward(self,x):

        for layers in self.layers:
            layer_output = layers(x)
            if self.use_shortcut: 
                x = x + layer_output
            else:
                x = layer_output
        return x
