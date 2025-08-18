

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports (same folder)
from dataset import MetricsMLPDataset
from network import metricsMLP


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------
# Train / Eval steps
# ---------------------------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits.detach(), yb) * bs
        n += bs
    return total_loss / n, total_acc / n


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, yb) * bs
            n += bs
    return total_loss / n, total_acc / n


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train an MLP classifier on metrics CSVs")

    # Data
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV: [trial_id | features... | label]")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV: same format as train")

    # Model
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)

    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # Datasets / Loaders
    train_ds = MetricsMLPDataset(args.train_csv)
    val_ds   = MetricsMLPDataset(args.val_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = metricsMLP(input_dim=train_ds.n_features,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                num_classes=train_ds.n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_epoch = -1
    ckpt_path = os.path.join(args.outdir, "mlp_best.pt")

    print(f"Device: {device}")
    print(f"Features: {train_ds.n_features} | Classes: {train_ds.n_classes}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "meta": {
                    "n_features": train_ds.n_features,
                    "n_classes": train_ds.n_classes,
                },
                "args": vars(args),
            }, ckpt_path)

    print(f"Best epoch: {best_epoch}  | best_val_loss: {best_val_loss:.4f}\nSaved: {ckpt_path}")


if __name__ == "__main__":
    main()