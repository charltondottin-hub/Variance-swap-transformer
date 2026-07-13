"""train_model for two-input (tuple) batches - composition, not edit.

The production loop in models/train.py calls x.to(device); our batches are
tuples of tensors, which have no .to(). Rather than edit production code on
an experiment branch, this module re-exports the loop with a two-line change
(_to below), reusing the loss and LR schedule from models.train verbatim.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.train import TrainConfig, qlike_loss_log, warmup_cosine_schedule


def _to(x, device):
    """Move a tensor or tuple-of-tensors to device."""
    if isinstance(x, (tuple, list)):
        return tuple(t.to(device) for t in x)
    return x.to(device)


def train_model(model: nn.Module, train_ds: Dataset, val_ds: Dataset,
                config: TrainConfig, init_as_baseline: bool = False):
    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    # Early surface segments have fewer total steps than the production
    # warmup (segment 1: 150 < 200), so the LR never finished ramping and
    # the cosine phase never ran. Cap warmup at 20% of the run.
    warmup_steps = min(config.warmup_steps, max(1, total_steps // 5))

    best_val = float("inf")
    best_state = None

    if init_as_baseline:               # warm-started weights are a candidate
        model.eval()
        with torch.no_grad():
            losses = [qlike_loss_log(model(_to(x, device)),
                                     y.to(device)).item()
                      for x, y in val_loader]
        best_val = sum(losses) / max(1, len(losses))
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Warm-start baseline | val {best_val:.5f}")

    bad_epochs, step = 0, 0
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = _to(x, device), y.to(device)

            lr_mult = warmup_cosine_schedule(step, warmup_steps,
                                             total_steps)
            for g in optimizer.param_groups:
                g["lr"] = config.lr * lr_mult

            loss = qlike_loss_log(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            step += 1

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                val_losses.append(qlike_loss_log(
                    model(_to(x, device)), y.to(device)).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / max(1, len(val_losses))
        print(f"Epoch {epoch + 1:3d} | train {train_loss:.5f} "
              f"| val {val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, None
