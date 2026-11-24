# LLM pretraining

To load the local GPT-2 checkpoint and generate the notebook's sample from the
repository root:

```bash
uv run --no-sync python -m notebooks.weights
```

The script uses the checkpoint in `notebooks/gpt2/124M` if present; otherwise it
downloads it using `notebooks/gpt_download.py`. It requires TensorFlow,
Requests, and tqdm in the active environment in addition to the project's
declared dependencies. The original notebook remains available with its saved
outputs.
