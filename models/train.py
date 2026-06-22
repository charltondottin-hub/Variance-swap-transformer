from dataclasses import dataclass, asdict
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

@dataclass
class TrainConfig: 
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int =30
    warmup_steps: int = 200
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    device: str = 'cpu'

def qlike_loss_log(log_pred: torch.Tensor, log_target: torch.Tensor) -> torch.Tensor:

    delta = log_target - log_pred
    return (torch.exp(delta) - delta - 1).mean()

def warmup_cosine_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainConfig,
):
    device = torch.device(config.device)
    model = model.to(device)
 
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
 
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    step = 0
 
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
 
            lr_mult = warmup_cosine_schedule(step, config.warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = config.lr * lr_mult
 
            pred = model(x)
            loss = qlike_loss_log(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
 
            train_losses.append(loss.item())
            step += 1
 
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_losses.append(qlike_loss_log(pred, y).item())
 
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / max(1, len(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
 
        print(f"Epoch {epoch + 1:3d} | train {train_loss:.5f} | val {val_loss:.5f}")
 
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
 
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
