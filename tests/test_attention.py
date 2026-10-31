import torch 

from attention.multi_head_attention import MultiHeadAttention


def test_mha_output_shape():
    torch.manual_seed(123)
    d_in = 3 
    d_out = 2
    batch = 2
    context_length = 6
    mha = MultiHeadAttention(d_in = d_in, d_out = d_out, context_length = context_length, 
                                 num_heads = 2, dropout = 0.0)
    x = torch.randn(batch, context_length, d_in)
    out = mha(x)
    assert out.shape == (batch, context_length, d_out)