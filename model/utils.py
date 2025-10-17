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

def run_training(train_ds, val_ds, args, device, outpath):
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

    best_val_loss = float("inf")
    best_epoch = -1
    ckpt_path = outpath

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
                "meta": {"n_features": train_ds.n_features,
                         "n_classes": train_ds.n_classes},
                "args": vars(args),
            }, ckpt_path)

    print(f"Best epoch: {best_epoch} | best_val_loss: {best_val_loss:.4f}\nSaved: {ckpt_path}")
    return best_val_loss, val_acc

def compute_permutation_importance(model, val_ds, criterion, device, feature_names, n_repeats=1):
    model.eval()
    X = val_ds.X.clone().to(device)
    y = val_ds.y.clone().to(device)
    num_samples = val_ds.n_samples

    # baseline
    with torch.no_grad():
        baseline_logits = model((X - val_ds.feature_mean.to(device)) / val_ds._std_safe.to(device))
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
                logits = model((X_perm - val_ds.feature_mean.to(device)) / val_ds._std_safe.to(device))
                acc = (logits.argmax(dim=1) == y).float().mean().item()
            drops.append(baseline_acc - acc)
        importances[feat_name] = sum(drops) / len(drops)
        print(f"{feat_name:40s}: Δacc = {importances[feat_name]:.4f}")

    return importances