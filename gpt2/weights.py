"""Load GPT-2 weights and generate a sample.

Run from the repository root with ``uv run python -m notebooks.weights``.
"""

import json
from pathlib import Path

import numpy as np
import tiktoken
import torch

from gpt2.config import GPT_CONFIG_124M
from gpt2.GPT2 import GPTModel
from gpt2.train import generate_text_with_topk, text_to_token_ids, token_ids_to_text

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"embed_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"embed_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"embed_dim": 1280, "n_layers": 36, "n_heads": 20},
}


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: model {left.shape}, checkpoint {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, dtype=left.dtype, device=left.device))


def load_weights_into_gpt(gpt, params):
    if len(gpt.trf_blocks) != len(params["blocks"]):
        raise ValueError("model and checkpoint have different numbers of transformer blocks")
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block, weights in zip(gpt.trf_blocks, params["blocks"], strict=True):
        attn = block.attn
        q_w, k_w, v_w = np.split(weights["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(weights["attn"]["c_attn"]["b"], 3, axis=-1)
        for projection, weight, bias in zip(
            (attn.W_q, attn.W_k, attn.W_v), (q_w, k_w, v_w), (q_b, k_b, v_b), strict=True
        ):
            if projection.bias is None:
                raise ValueError("model must be created with qkv_bias=True")
            projection.weight = assign(projection.weight, weight.T)
            projection.bias = assign(projection.bias, bias)

        attn.out_proj.weight = assign(attn.out_proj.weight, weights["attn"]["c_proj"]["w"].T)
        attn.out_proj.bias = assign(attn.out_proj.bias, weights["attn"]["c_proj"]["b"])

        block.ff.layers[0].weight = assign(block.ff.layers[0].weight, weights["mlp"]["c_fc"]["w"].T)
        block.ff.layers[0].bias = assign(block.ff.layers[0].bias, weights["mlp"]["c_fc"]["b"])
        block.ff.layers[2].weight = assign(block.ff.layers[2].weight, weights["mlp"]["c_proj"]["w"].T)
        block.ff.layers[2].bias = assign(block.ff.layers[2].bias, weights["mlp"]["c_proj"]["b"])

        block.norm1.scale = assign(block.norm1.scale, weights["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, weights["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, weights["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, weights["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def main():
    import tensorflow as tf

    from notebooks.gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt

    models_dir = Path(__file__).resolve().parent / "gpt2"
    model_dir = models_dir / "124M"
    checkpoint_path = tf.train.latest_checkpoint(str(model_dir))
    if checkpoint_path is None:
        settings, params = download_and_load_gpt2(model_size="124M", models_dir=str(models_dir))
    else:
        with (model_dir / "hparams.json").open(encoding="utf-8") as file:
            settings = json.load(file)
        params = load_gpt2_params_from_tf_ckpt(checkpoint_path, settings)

    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)

    config = GPT_CONFIG_124M.copy()
    config.update(MODEL_CONFIGS["gpt2-small (124M)"])
    config.update({"context_length": 1024, "qkv_bias": True})
    model = GPTModel(config).eval()
    load_weights_into_gpt(model, params)
    device = torch.device("cpu")
    model.to(device)

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_with_topk(
        model=model,
        idx=text_to_token_ids("Hello I am Bilbo", tokenizer).to(device),
        max_token_length=25,
        context_size=config["context_length"],
        k=50,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
