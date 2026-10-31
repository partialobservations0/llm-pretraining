import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout_rate = 0.1

        self.W_q = torch.nn.Linear(d_in, d_out)
        self.W_k = torch.nn.Linear(d_in, d_out)
        self.W_v = torch.nn.Linear(d_in, d_out)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length),diagonal=1))
        
    def forward(self, x):

        _b, num_tokens, _d_in = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = torch.matmul(q, k.transpose(1,2))
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weight = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)
        context_vec = attn_weight @ v
        return context_vec


inputs = torch.tensor([
    [0.41, 0.2, 0.89],
    [0.55, 0.83, 0.66], 
    [0.57, 0.85, 0.64],
    [0.2, 0.58, 0.3],
    [0.77, 0.24, 0.1], 
    [0.05, 0.80, 0.55]
    ])

if __name__ == '__main__':

    batch = torch.stack((inputs, inputs), dim=0)
    logger.info(f"batch size {batch.shape}")

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(3, 2, context_length)
    context_vec = ca(batch)
    logger.info(f"context vec {context_vec}")
