import tiktoken
import torch

from gpt2.GPT2 import GPT_CONFIG_124M, GPTModel
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
        self.pos_emb = torch.nn.Embedding(3, 1)

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

def temperature_logits():
    config = {
        "vocab_size": 50257,
        "context_length": 32,
        "embed_dim": 8,
        "n_heads": 2,
        "n_layers": 1,
        "dropout_rate": 0.0,
        "qkv_bias": False,
    }
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = GPTModel(config).eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Explain quantum computing to me"
    ids = text_to_token_ids(text, tokenizer)
    with torch.no_grad():
        logits = model(ids)[0, -1, :]

    return torch.softmax(logits / 5.0,dim=-1), torch.softmax(logits / 0.5,dim=-1)


def plot_temperature_logits(temp50_logits, temp05_logits):
    import matplotlib.pyplot as plt

    top_ids = temp05_logits.topk(20).indices
    labels = [str(token_id.item()) for token_id in top_ids]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(labels, temp50_logits[top_ids].cpu().numpy(), marker="o", label="Temperature 5.0")
    ax.plot(labels, temp05_logits[top_ids].cpu().numpy(), marker="o", label="Temperature 0.5")
    ax.set_xlabel("Token ID (top 20 by logit)")
    ax.set_ylabel("Scaled logit")
    ax.set_title("Temperature-scaled logits (randomly initialized model)")
    ax.tick_params(axis="x", labelrotation=60)
    ax.legend()
    fig.tight_layout()
    return fig


def test_generation_with_temperature():
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    text = "Explain quantum computing to me"
    ids = text_to_token_ids(text, tokenizer)
    logits = model(ids)
    #probs = torch.softmax(logits[:,-1,:],dim=-1)

    temp = 5.0
    if temp > 0:
        temp_logits = logits / temp
    else:
        temp_logits = logits
    temp_logits = logits / temp
    temp50_probs = torch.softmax(temp_logits[:,-1,:],dim=-1)

    temp = 0.5
    if temp > 0:
        temp_logits = logits / temp
    else:
        temp_logits = logits
    temp05_probs = torch.softmax(temp_logits[:,-1,:], dim=-1)
    return temp05_probs, temp50_probs

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    temp50_logits, temp05_logits = temperature_logits()
    plot_temperature_logits(temp50_logits, temp05_logits)
    plt.show()
