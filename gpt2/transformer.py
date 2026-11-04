import logging

import torch

from attention.multi_head_attention import MultiHeadAttention
from gpt2.config import GPT_CONFIG_124M
from gpt2.dummy_gpt import DummyLayerNorm, MyFeedForward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerBlock(torch.nn.Module):

    def __init__(self, cfg):

        super().__init__()
        self.attn = MultiHeadAttention(d_in = cfg["embed_dim"],
                                       d_out = cfg["embed_dim"],
                                       context_length=cfg["context_length"],
                                       num_heads=cfg["n_heads"],
                                       dropout=cfg["dropout_rate"],
                                       qkv_bias=cfg["qkv_bias"]
                                       )
        self.ff = MyFeedForward(cfg=cfg)
        self.norm1 = DummyLayerNorm(embed_dim=cfg["embed_dim"])
        self.norm2 = DummyLayerNorm(embed_dim=cfg["embed_dim"])
        self.drop_shortcut = torch.nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"input shape {x.shape}")
print(f"output shape {output.shape}")

attn_params_count = sum([p.numel() for p in block.attn.parameters()])
print(f"attn parameters count: {attn_params_count}")
ffn_params_count = sum(p.numel() for p in block.ff.parameters())
print(f"ff params count: {ffn_params_count}")