import subprocess
import sys

import pytest

MODULES = [
    "attention.causal_attn",
    "attention.multi_head_attention",
    "attention.self_attn",
    "attention.simplified",
    "embeddings.Tokenizer",
    "embeddings.embeddings",
    "embeddings.tiktoken_tokenizer",
    "embeddings.words",
    "gpt2.dummy_gpt",
    "gpt2.transformer",
    "gpt2.GPT2",
    "gpt2.train",
]


@pytest.mark.parametrize("configured", [False, True])
def test_imports_are_quiet_and_do_not_configure_root_logging(configured):
    code = f"""
import importlib
import logging
import torch

if {configured!r}:
    logging.basicConfig(level=logging.INFO)
root = logging.getLogger()
before_handlers = root.handlers[:]
before_level = root.level
torch.manual_seed(42)
before_rng = torch.random.get_rng_state()
for module in {MODULES!r}:
    importlib.import_module(module)
assert root.handlers == before_handlers
assert root.level == before_level
assert torch.equal(torch.random.get_rng_state(), before_rng)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("attention.causal_attn", "batch size"),
        ("attention.self_attn", "causal_attn_scores"),
        ("attention.simplified", "logger ready"),
        ("gpt2.transformer", "output shape"),
    ],
)
def test_running_module_still_outputs_demo(module, expected):
    result = subprocess.run([sys.executable, "-m", module], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout + result.stderr
