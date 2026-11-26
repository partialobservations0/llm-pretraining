import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetuning.tune import calc_accuracy_loader, calc_loss_loader, train_classifier_simple


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_head = torch.nn.Linear(1, 2)

    def forward(self, inputs):
        return self.out_head(inputs.float().unsqueeze(-1))


def test_training_loop_uses_validation_and_updates_head():
    torch.manual_seed(123)
    model = TinyClassifier()
    train = DataLoader(TensorDataset(torch.tensor([[0], [1]]), torch.tensor([0, 1])), batch_size=2)
    validation = DataLoader(
        TensorDataset(torch.tensor([[1], [1]]), torch.tensor([1, 1])), batch_size=2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    old_weights = model.out_head.weight.detach().clone()
    losses, val_losses, accuracies, val_accuracies, examples = train_classifier_simple(
        model, train, validation, optimizer, "cpu", num_epochs=1, eval_freq=1, eval_iter=1
    )
    assert examples == 2
    assert len(losses) == len(val_losses) == len(accuracies) == len(val_accuracies) == 1
    assert 0 <= accuracies[0] <= 1
    assert 0 <= val_accuracies[0] <= 1
    assert not torch.equal(old_weights, model.out_head.weight)


def test_empty_loaders_fail_explicitly():
    model = TinyClassifier()
    with pytest.raises(ValueError, match="without examples"):
        calc_accuracy_loader([], model)
    with pytest.raises(ValueError, match="empty"):
        calc_loss_loader([], model, "cpu")


def test_accuracy_uses_all_test_batches():
    class FixedClassifier(torch.nn.Module):
        def forward(self, inputs):
            return torch.nn.functional.one_hot(inputs, num_classes=2).float()

    loader = DataLoader(
        TensorDataset(torch.tensor([[0], [1], [1]]), torch.tensor([0, 0, 1])),
        batch_size=2,
    )
    assert calc_accuracy_loader(loader, FixedClassifier()) == pytest.approx(2 / 3)
