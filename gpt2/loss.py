import tiktoken
import torch
from torch.nn.utils.rnn import pad_sequence

from gpt2.GPT2 import GPT_CONFIG_124M, GPTModel


def generate_text(model, idx, max_token_length, context_size):
    # idx is batch of token ids of size batch x seq length
    for _ in range(max_token_length):
        context = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(context)
        #print(logits.shape) # logits shape is batch x context_size x vocab_size
        #note: the context dimension is because model produces next-token for every word in -context: seq
        #to generate new text we only need the latest word decoding
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        tokens = torch.argmax(probs, dim=-1, keepdim=True)
        #print(tokens.shape)
        idx = torch.cat([context,tokens],dim=1)
    return None

def generate_text_for_batched_input(model, encsequences, max_token_length, context_size):
    """
    Main point is that for batched input of different length we will have to use
    zero padding and then extract the argmax for logits that belong to first non-zero
    token of each text
    """
    model.eval()
    with torch.no_grad():
        contexts = [seq[-context_size:] for seq in encsequences]
        lengths = torch.tensor([len(s) for s in contexts])
        batch = pad_sequence(contexts, batch_first=True, padding_side='right')
        logits = model(batch)
        # we want to select for each row, position = length[row] from logits
        last_logits = logits[torch.arange(len(lengths)), lengths-1] # no need for[ , ,: ] its implicit
        probs = torch.softmax(last_logits, dim=-1)
        tokens = torch.argmax(probs, dim=-1, keepdim=True)
        encsequences = [
                        torch.cat([seq, token.view(1)])
                        for seq, token in zip(encsequences, tokens, strict=False)
                        ]

        return encsequences



def test_generate_text():
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding('gpt2')
    text = "Explain quantum computing to a layman:"
    idx = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    generate_text(model, idx, 4, 3)

def test_generate_text_for_batched_input():
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding('gpt2')
    text = [["Explain quantum computing to a layman:"], ["Explain blockchain to a layman:"]]
    encseq = [torch.tensor(tokenizer.encode(item[0])) for item in text]
    generate_text_for_batched_input(model, encseq, 4, 3 )
        

test_generate_text()
test_generate_text_for_batched_input()

