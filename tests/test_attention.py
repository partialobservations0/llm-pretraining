import pytest
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


def _copy_weights_to_reference(mha: MultiHeadAttention) -> torch.nn.MultiheadAttention:
    """Build nn.MultiheadAttention with the same weights as our module.

    nn.MultiheadAttention stores Q, K, V as one stacked (3*d, d) matrix `in_proj_weight`.
    Its `bias` flag covers both the in-projection and out_proj, so we keep bias=True,
    zero the in-projection bias (our qkv_bias=False) and copy out_proj's bias.
    """
    ref = torch.nn.MultiheadAttention(
        embed_dim=mha.d_out, num_heads=mha.num_heads, dropout=0.0, bias=True, batch_first=True
    )
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
        ref.in_proj_bias.zero_()
        ref.out_proj.weight.copy_(mha.out_proj.weight)
        ref.out_proj.bias.copy_(mha.out_proj.bias)
    return ref.eval()


@pytest.mark.parametrize("num_tokens", [6, 4])  # full context and a shorter sequence
def test_matches_torch_multihead_attention(num_tokens):
    torch.manual_seed(0)
    d, num_heads, context_length, batch = 8, 2, 6, 3  # nn.MHA needs d_in == d_out
    mha = MultiHeadAttention(d, d, context_length, num_heads, dropout=0.0).eval()
    ref = _copy_weights_to_reference(mha)

    x = torch.randn(batch, num_tokens, d)
    causal_mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)

    with torch.no_grad():
        ours = mha(x)
        theirs, _ = ref(x, x, x, attn_mask=causal_mask, need_weights=False)

    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


def test_causality_future_tokens_do_not_affect_past():
    torch.manual_seed(1)
    d_in, d_out, context_length, t = 3, 4, 6, 3
    mha = MultiHeadAttention(d_in, d_out, context_length, num_heads=2, dropout=0.0).eval()

    x = torch.randn(2, context_length, d_in)
    x_perturbed = x.clone()
    x_perturbed[:, t:, :] += 10 * torch.randn_like(x_perturbed[:, t:, :])

    with torch.no_grad():
        out = mha(x)
        out_perturbed = mha(x_perturbed)

    # positions < t only see tokens < t, so they must be identical
    torch.testing.assert_close(out[:, :t], out_perturbed[:, :t])
    # positions >= t do see the change (guards against a test that passes trivially)
    assert not torch.allclose(out[:, t:], out_perturbed[:, t:])


def test_shapes_d_out_4_two_heads():
    torch.manual_seed(2)
    d_in, d_out, num_heads, context_length, batch = 3, 4, 2, 6, 2
    mha = MultiHeadAttention(d_in, d_out, context_length, num_heads, dropout=0.0)

    assert mha.head_dim == 2
    for proj in (mha.W_q, mha.W_k, mha.W_v):
        assert proj.weight.shape == (d_out, d_in)
    assert mha.out_proj.weight.shape == (d_out, d_out)
    assert mha.mask.shape == (context_length, context_length)

    for num_tokens in (context_length, 1):  # full and shorter-than-context input
        out = mha(torch.randn(batch, num_tokens, d_in))
        assert out.shape == (batch, num_tokens, d_out)


def test_dropout_eval_vs_train():
    torch.manual_seed(3)
    d_in, d_out, context_length = 3, 4, 6
    mha = MultiHeadAttention(d_in, d_out, context_length, num_heads=2, dropout=0.5)
    no_drop = MultiHeadAttention(d_in, d_out, context_length, num_heads=2, dropout=0.0)
    no_drop.load_state_dict(mha.state_dict())
    x = torch.randn(2, context_length, d_in)

    # eval: dropout is a no-op, so output is deterministic and equals the dropout=0 model
    mha.eval()
    no_drop.eval()
    with torch.no_grad():
        e1, e2, ref = mha(x), mha(x), no_drop(x)
    torch.testing.assert_close(e1, e2)
    torch.testing.assert_close(e1, ref)

    # train: dropout is active, so outputs are stochastic and differ from eval
    mha.train()
    with torch.no_grad():
        t1, t2 = mha(x), mha(x)
    assert not torch.allclose(t1, t2)
    assert not torch.allclose(t1, e1)
