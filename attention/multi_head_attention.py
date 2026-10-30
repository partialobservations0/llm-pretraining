import torch
from causal_attn import CausalAttention, inputs

class WrapperMultiHeadAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length) for _ in range(num_heads)]
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):

        super().__init__()
        self.d_in, self.d_out, self.context_length = d_in, d_out, context_length
        self.hum_heads = num_heads, dropout
        assert (d_out % num_heads == 0)
        self.head_dim = d_out // num_heads
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        b, num_tokens, d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        # torch transpose function 







if __name__ == '__main__':

    batch = torch.stack([inputs, inputs], dim=0)
    torch.manual_seed(121)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    mha = WrapperMultiHeadAttention(d_in, d_out, context_length, 0.5, 2)
    context_vec = mha(batch)
    print(context_vec)
