import torch

from gpt2.transformer import TransformerBlock
from gpt2.dummy_gpt import LayerNorm
from gpt2.config import GPT_CONFIG_124M

class GPTModel(torch.nn.Module):

    def __init__(self,cfg):

        super().__init__()
        self.tok_embedding = torch.nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_embedding = torch.nn.Embedding(cfg["context_length"], cfg["embed_dim"])   
        self.drop_emb = torch.nn.Dropout(cfg["dropout_rate"])
        self.trf_blocks = torch.nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["embed_dim"])
        self.out_head = torch.nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_embedding(in_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
batch = torch.tensor([[6109, 3626, 6100, 345],
                       [6109, 1110, 6622, 257]])
out = model(batch)
print(out.shape)

total_params = sum([p.numel() for p in model.parameters()])
print(f"total parameters {total_params}")
total_params_in_trf = sum(p.numel() for p in model.trf_blocks.parameters())
print(f"total parameters {total_params_in_trf}")