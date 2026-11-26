
import time

import tiktoken
import torch
from torch.utils.data import DataLoader

from finetuning.dataset import SpamDataset
from gpt2.train import generate_and_print_sample
from gpt2.weights import load_pretrained_gpt2

device = 'cpu'
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "drop_rate": 0.0,
 "qkv_bias": True
}
model_configs = {
 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

def calc_accuracy_loader(data_loader, model, device='cpu', num_batches=None):
    model.eval()

    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_label = torch.argmax(logits, dim=-1)
            num_examples += predicted_label.shape[0]
            correct_predictions += ((predicted_label == target_batch).sum().item())
        else:
            break
    if num_examples == 0:
        raise ValueError("Cannot calculate accuracy without examples")
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device=device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader,model,device,num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        raise ValueError("Cannot calculate loss for an empty data loader")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_classifier_simple(model, train_loader, validation_loader,
                            optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_acc, val_acc = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, validation_loader,
                                                      device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} Step {global_step} Train_loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        train_accuracy = calc_accuracy_loader(train_loader,model,device,num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(validation_loader,model,device,num_batches=eval_iter)
        print(f"Training accuracy {train_accuracy}")
        print(f"validation accuracy {val_accuracy}")
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
    return train_losses, val_losses, train_acc, val_acc, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

def main():
    torch.manual_seed(123)
    loc = './notebooks/'
    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = SpamDataset(csv_file=loc + 'train.csv', tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file=loc + 'test.csv', tokenizer=tokenizer,
                               max_length=train_dataset.max_length)
    validation_dataset = SpamDataset(csv_file=loc + 'validation.csv', tokenizer=tokenizer,
                                     max_length=train_dataset.max_length)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=8)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=8)
    validation_loader = DataLoader(validation_dataset, shuffle=False, num_workers=0, batch_size=8)
    print(f"len training batches {len(train_loader)}")

    model = load_pretrained_gpt2()
    generate_and_print_sample(model=model, tokenizer=tokenizer,
                              start_context="Explain quantum mechanics", device=device)
    for params in model.parameters():
        params.requires_grad = False
    model.out_head = torch.nn.Linear(in_features=model.out_head.in_features, out_features=2)
    for params in model.trf_blocks[-1].parameters():
        params.requires_grad = True
    for params in model.final_norm.parameters():
        params.requires_grad = True
    model.to(device)
    torch.manual_seed(123)
    for name, loader in (("Training", train_loader), ("Validation", validation_loader),
                         ("Test", test_loader)):
        accuracy = calc_accuracy_loader(loader, model, device, num_batches=10)
        print(f"{name} accuracy before training: {accuracy*100:.2f}%")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=5e-5, weight_decay=0.1
    )
    train_classifier_simple(model, train_loader, validation_loader, optimizer, device,
                            num_epochs=1, eval_freq=50, eval_iter=5)
    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Test accuracy after training: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
