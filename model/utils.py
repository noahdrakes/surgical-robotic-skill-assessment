import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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


def plot_confusion_matrix(y_true, y_pred, labels, class_names, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()




# ----------------------------
# Train/Eval
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
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


def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, yb) * bs
            n += bs
            if return_preds:
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(yb.cpu().numpy())
    if return_preds:
        return total_loss / n, total_acc / n, np.array(all_preds), np.array(all_targets)
    else:
        return total_loss / n, total_acc / n


# ---------------------------
# Training wrapper
# ---------------------------

def run_training(train_ds, val_ds, args, device, outpath, return_history=False):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = metricsMLP(input_dim=train_ds.n_features,
                       hidden1=args.hidden1,
                       hidden2=args.hidden2,
                       num_classes=train_ds.n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    prev_lr = optimizer.param_groups[0]["lr"]

    best_val_loss = float("inf")
    best_epoch = -1
    ckpt_path = outpath

    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            wd = optimizer.param_groups[0].get("weight_decay", 0.0)
            print(f"Epoch {epoch:03d} | lr decayed to {new_lr:.2e} | weight_decay: {wd:.2e}")
        prev_lr = new_lr
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # print(
        #     f"Epoch {epoch:03d} | "
        #     f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
        #     f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
        #     f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        # )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "meta": {"n_features": train_ds.n_features,
                         "n_classes": train_ds.n_classes},
                "args": vars(args),
            }, ckpt_path)

    print(f"Best epoch: {best_epoch} | best_val_loss: {best_val_loss:.4f}\nSaved: {ckpt_path}")
    if return_history:
        history = {"train_loss": train_losses, "val_loss": val_losses}
        return best_val_loss, val_acc, history
    return best_val_loss, val_acc



#### PERMUTATION IMPORTANCE #########

def compute_permutation_importance(model, dataset, criterion, device, feature_names, n_repeats=1):
    model.eval()
    X = dataset.X.clone().to(device)
    y = dataset.y.clone().to(device)
    num_samples = dataset.n_samples

    # baseline
    with torch.no_grad():

        # pass dataset normalized dataset through model to get baseline accuracy
        baseline_logits = model((X - dataset.feature_mean.to(device)) / dataset._std_safe.to(device))
        baseline_acc = (baseline_logits.argmax(dim=1) == y).float().mean().item()

    print(f"\nBaseline accuracy: {baseline_acc:.4f}")

    importances = {}
    for feat_idx, feat_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.clone()
            perm_idx = torch.randperm(num_samples)
            X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
            with torch.no_grad():
                logits = model((X_perm - dataset.feature_mean.to(device)) / dataset._std_safe.to(device))
                acc = (logits.argmax(dim=1) == y).float().mean().item()
            drops.append(baseline_acc - acc)
        importances[feat_name] = sum(drops) / len(drops)
        print(f"{feat_name:40s}: Δacc = {importances[feat_name]:.4f}")

    return importances


def plot_permutation_importance(importances, title="Permutation Importance (Δ Accuracy)"):
    # Convert to sorted list of tuples
    features, values = zip(*sorted(importances.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(8, 6))
    plt.barh(features, values, color="skyblue", edgecolor="black")
    plt.gca().invert_yaxis()  # most important feature at top
    plt.xlabel("Δ Accuracy (drop after shuffling)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curves(train_losses, val_losses, outpath, title="Training vs Validation Loss"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
