import numpy as np
import pytest
import torch
from notebooks.weights import load_weights_into_gpt

from gpt2.GPT2 import GPTModel


def test_loads_checkpoint_parameters_and_runs_model():
    d, vocab, context = 4, 9, 8
    config = {
        "vocab_size": vocab,
        "context_length": context,
        "embed_dim": d,
        "n_heads": 2,
        "n_layers": 1,
        "dropout_rate": 0.0,
        "qkv_bias": True,
    }
    rng = np.random.default_rng(42)

    def array(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    block = {
        "attn": {
            "c_attn": {"w": array(d, 3 * d), "b": array(3 * d)},
            "c_proj": {"w": array(d, d), "b": array(d)},
        },
        "mlp": {
            "c_fc": {"w": array(d, 4 * d), "b": array(4 * d)},
            "c_proj": {"w": array(4 * d, d), "b": array(d)},
        },
        "ln_1": {"g": array(d), "b": array(d)},
        "ln_2": {"g": array(d), "b": array(d)},
    }
    params = {
        "blocks": [block], "wpe": array(context, d), "wte": array(vocab, d),
        "g": array(d), "b": array(d),
    }
    model = GPTModel(config).eval()
    load_weights_into_gpt(model, params)
    attn = model.trf_blocks[0].attn
    for projection, weight, bias in zip(
        (attn.W_q, attn.W_k, attn.W_v),
        np.split(block["attn"]["c_attn"]["w"], 3, axis=-1),
        np.split(block["attn"]["c_attn"]["b"], 3),
        strict=True,
    ):
        torch.testing.assert_close(projection.weight, torch.from_numpy(weight.T))
        torch.testing.assert_close(projection.bias, torch.from_numpy(bias))
    torch.testing.assert_close(model.out_head.weight, torch.from_numpy(params["wte"]))
    torch.testing.assert_close(
        model.trf_blocks[0].ff.layers[0].weight, torch.from_numpy(block["mlp"]["c_fc"]["w"].T)
    )
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, vocab)
    assert torch.isfinite(logits).all()

    with pytest.raises(ValueError, match="different numbers"):
        load_weights_into_gpt(model, {**params, "blocks": []})
