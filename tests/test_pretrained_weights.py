import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gpt2 import weights
from gpt2.GPT2 import GPTModel
from gpt2.weights import load_pretrained_gpt2, load_weights_into_gpt


@pytest.mark.parametrize("has_checkpoint", [True, False])
def test_load_pretrained_gpt2_uses_checkpoint_or_downloader(tmp_path, monkeypatch, has_checkpoint):
    checkpoint_dir = tmp_path / "124M"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "hparams.json").write_text(json.dumps({"n_layer": 12}))
    checkpoint = str(checkpoint_dir / "model.ckpt")
    params = object()
    calls = []

    def latest_checkpoint(path):
        assert path == str(checkpoint_dir)
        return checkpoint if has_checkpoint else None

    def load_checkpoint(path, settings):
        assert path == checkpoint
        assert settings == {"n_layer": 12}
        calls.append("local")
        return params

    def download(model_size, models_dir):
        assert model_size == "124M"
        assert models_dir == str(tmp_path)
        calls.append("download")
        return {"n_layer": 12}, params

    monkeypatch.setitem(
        sys.modules, "tensorflow", SimpleNamespace(train=SimpleNamespace(latest_checkpoint=latest_checkpoint))
    )
    monkeypatch.setitem(
        sys.modules,
        "notebooks.gpt_download",
        SimpleNamespace(
            download_and_load_gpt2=download,
            load_gpt2_params_from_tf_ckpt=load_checkpoint,
        ),
    )
    model = torch.nn.Linear(1, 1)

    def make_model(config):
        assert config["qkv_bias"] is True
        assert config["context_length"] == 1024
        return model

    def load_weights(loaded_model, loaded_params):
        assert loaded_model is model
        assert loaded_params is params

    monkeypatch.setattr(weights, "GPTModel", make_model)
    monkeypatch.setattr(weights, "load_weights_into_gpt", load_weights)
    assert load_pretrained_gpt2(tmp_path) is model
    assert not model.training
    assert calls == ["local" if has_checkpoint else "download"]


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
