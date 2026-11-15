import tiktoken
import torch
from path import Path
from torch.nn.utils.rnn import pad_sequence

from embeddings.tiktoken_tokenizer import create_dataloader_v2
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
        idx = torch.cat([idx, tokens], dim=1)
    return idx

def generate_text_with_topk(model, idx, max_token_length, context_size, k=10):

    for _ in range(max_token_length):
        context = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(context)
        logits = logits[:,-1,:]
        # logits is batch x seq_length x vocab
        top_logits, _top_pos = torch.topk(input=logits, k=k, dim=-1,)
        new_logits = torch.where(logits >= top_logits[..., -1:], logits, other=-1*torch.inf)
        probs = torch.softmax(new_logits, dim=-1)
        tokens = torch.multinomial(probs,num_samples=1)
        idx = torch.cat([idx, tokens], dim=1)
    return idx

def generate_text_temperature_scaled(model, idx, max_token_length, context_size, temp=1.0):

    for _ in range(max_token_length):
        context = idx[:, - context_size:]
        with torch.no_grad():
            logits = model(context)
        logits = logits / temp
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        tokens = torch.multinomial(input=probs,num_samples=1).item()
        idx = torch.cat([idx, tokens], dim=1)
        # item returns the value of tensor as a python number
    return idx

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

def get_train_test_data():
    root_dir = Path.cwd()
    print(root_dir)
    filepath = root_dir + '/embeddings/the-verdict.txt'
    with open(filepath) as f:
        text = f.read()
    n = int(0.9 * len(text))
    train_text, val_text = text[0:n], text[n:]
    return train_text, val_text

def get_data_loaders(train_text, val_text):
    tokenizer = tiktoken.get_encoding('gpt2')
    train_loader = create_dataloader_v2(tokenizer, train_text, batch_size=16, max_length=8,
                                        stride=4, shuffle=True, drop_last=True, num_workers=2)
    val_loader = create_dataloader_v2(tokenizer, val_text, batch_size=16, max_length=8,
                                        stride=4, shuffle=False, drop_last=True, num_workers=2)
    return train_loader, val_loader

def calc_loss_batch(model, input_batch, target_batch, device='cpu'):
    logits = model(input_batch)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def train_simple_model(model, train_loader, val_loader,tokenizer, optimizer, device='cpu', 
                    num_epochs=20, eval_freq=10, eval_iter=5, start_context="Hello I am Bilbo"):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(model, input_batch, target_batch)
            loss.backward()
            optimizer.step()   
            global_step += 1
            token_seen += input_batch.numel()
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss: {train_loss:.3f}"
                    f"Val loss: {val_loss:.3f}"
                    )
                generate_and_print_sample(model, tokenizer, device, start_context)
                save_model(model,optimizer)

    return train_losses, val_losses, track_tokens_seen

def calc_loss_loader(dataloader, model, device, num_batches = None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):

    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_with_topk(model = model, idx = encoded, max_token_length = 50, context_size = context_size)
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())

def save_model(model,optimizer):
    model.eval()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()}, "model_and_optimizer.pth")
    
def load_model():
    checkpoint = torch.load("model_and_optimizer.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    optimizer = optimizer.load_state_dict(checkpoint["optimizer"])
    model.train()
    print("model loaded")

if __name__ == '__main__':
    train_text, val_text = get_train_test_data()
    train_loader, val_loader = get_data_loaders(train_text, val_text)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.Adam(model.parameters())
    train_losses, val_losses, track_tokens_seen = train_simple_model(model, train_loader, val_loader, tokenizer, optimizer)

