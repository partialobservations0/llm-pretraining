# LLM pretraining

To load pretrained 124M GPT-2 weights into the project model:

```python
from gpt2.weights import load_pretrained_gpt2

model = load_pretrained_gpt2()
```

The function returns a model in evaluation mode on the CPU. You can provide
`models_dir` to use a different checkpoint directory; the checkpoint is expected
in its `124M` subdirectory. To generate the notebook's sample from the repository
root:

```bash
uv run --no-sync python -m gpt2.weights
```

The script uses the checkpoint in `notebooks/gpt2/124M` if present; otherwise it
downloads it using `notebooks/gpt_download.py`. It requires TensorFlow,
Requests, and tqdm in the active environment in addition to the project's
declared dependencies. The original notebook remains available with its saved
outputs.
