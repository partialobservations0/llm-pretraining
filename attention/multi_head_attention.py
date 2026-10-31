import torch
from attention.causal_attn import CausalAttention, inputs

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
        self.num_heads = num_heads
        assert (d_out % num_heads == 0)
        self.head_dim = d_out // num_heads
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # here diagonal = 1 means start from one diagonal above the main diagonal


    def forward(self, x):

        b, num_tokens, d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # swap dimension 1 with 2 resulting in shape b, num_heads, num_tokens, head_dim
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # torch transpose function and why this transpose ??

        attn_scores = torch.matmul(queries, keys.transpose(2,3))
        # what would be attn_scores dimensions here? b, num_heads, num_tokens, num_tokens 
        # also explains the earlier transpose.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # masked_fill will write -torch.inf where it has value True (triu)  
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = torch.matmul(attn_weights, values).transpose(1,2).view(b,num_tokens,self.d_out)
        return context_vec



if __name__ == '__main__':

    batch = torch.stack([inputs, inputs], dim=0)
    torch.manual_seed(121)
    batch_size, context_length, d_in = batch.shape
    d_in, d_out = 3, 2
    #mha = WrapperMultiHeadAttention(d_in, d_out, context_length, 0.5, 2)
    mha = MultiHeadAttention(d_in, d_out, context_length, 2, 0.5)
    context_vec = mha(batch)
    print(context_vec)
