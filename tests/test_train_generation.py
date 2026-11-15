import tiktoken
import torch

from gpt2.GPT2 import GPTModel
from gpt2.train import (
    evaluate_model,
    generate_and_print_sample,
    generate_text,
    text_to_token_ids,
    token_ids_to_text,
    train_simple_model,
)


class NextTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = torch.nn.Embedding(3, 1)

    def forward(self, token_ids):
        logits = torch.zeros(*token_ids.shape, 50257, device=token_ids.device)
        logits[..., 0] = 1
        return logits


def test_token_conversion_and_generation_keep_prompt():
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello I am Bilbo"
    ids = text_to_token_ids(prompt, tokenizer)
    assert ids.shape == (1, len(tokenizer.encode(prompt)))
    assert token_ids_to_text(ids, tokenizer) == prompt

    result = generate_text(NextTokenModel(), ids, max_token_length=4, context_size=3)
    assert result.shape == (1, ids.shape[1] + 4)
    torch.testing.assert_close(result[:, :ids.shape[1]], ids)


def test_generate_and_print_sample(capsys):
    tokenizer = tiktoken.get_encoding("gpt2")
    model = NextTokenModel()
    generate_and_print_sample(model, tokenizer, "cpu", "Hello")
    assert "Hello" in capsys.readouterr().out
    assert model.training


def test_training_reaches_sample_generation(capsys):
    config = {
        "vocab_size": 50257,
        "context_length": 8,
        "embed_dim": 8,
        "n_heads": 2,
        "n_layers": 1,
        "dropout_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(config)
    batch = (torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.Adam(model.parameters())
    losses, val_losses, tokens = train_simple_model(
        model, [batch], [batch], tokenizer, optimizer,
        num_epochs=1, eval_freq=1, eval_iter=1, start_context="Hello",
    )
    assert len(losses) == len(val_losses) == 1
    assert tokens == [2]
    assert "Hello" in capsys.readouterr().out


def test_evaluate_uses_all_validation_batches():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, token_ids):
            self.calls += 1
            return torch.zeros(*token_ids.shape, 3)

    model = CountingModel()
    batch = (torch.tensor([[1]]), torch.tensor([[2]]))
    evaluate_model(model, [batch] * 8, [batch] * 8, "cpu", eval_iter=5)
    assert model.calls == 5 + 8
    assert model.training
