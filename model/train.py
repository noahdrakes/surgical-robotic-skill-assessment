import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Local imports
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


# ---------------------------
# Train / Eval steps
# ---------------------------

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


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train an MLP classifier on metrics CSVs")

    # Data
    parser.add_argument("--train_csv", type=str, help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, help="Path to validation CSV")
    parser.add_argument("--all_csv", type=str, help="Path to CSV with all subjects (for LOUO mode)")
    parser.add_argument("--validation_mode", type=str, default="split",
                        choices=["split", "louo", "loso"],
                        help="Validation mode: 'split' (manual CSVs), 'louo' (leave-one-user-out), or 'loso' (leave-one-supertrial-out)")

    ## FORCE METRICS
    FORCE_FEATURES = [
        "ATIForceSensor_average_force_magnitude",
        "PSM1_forcen_magnitude",
        "PSM2_forcen_magnitude"
    ]

    FORCE_ACCEL_FEATURES = [
        "ATIForceSensor_average_force_magnitude",
        "PSM1_forcen_magnitude",
        "PSM2_forcen_magnitude",
        "PSM1_average_acceleration_magnitude",
        "PSM2_average_acceleration_magnitude",
        "PSM1_PSM2_acceleration_cross",
        "PSM1_PSM2_jerk_dispertion",
    ]

    ALL_FEATURES = [
        "completion_time",
        "PSM1_average_speed_magnitude",
        "PSM2_average_speed_magnitude",
        "PSM1_average_acceleration_magnitude",
        "PSM2_average_acceleration_magnitude",
        "PSM1_average_jerk_magnitude",
        "PSM2_average_jerk_magnitude",
        "ATIForceSensor_average_force_magnitude",
        "PSM1_total_path_length",
        "PSM2_total_path_length",
        "PSM1_average_angular_speed_magnitude",
        "PSM2_average_angular_speed_magnitude",
        "PSM1_PSM2_speed_correlation",
        "PSM1_PSM2_speed_cross",
        "PSM1_PSM2_acceleration_cross",
        "PSM1_PSM2_jerk_cross",
        "PSM1_PSM2_acceleration_dispertion",
        "PSM1_PSM2_jerk_dispertion",
        "PSM1_forcen_magnitude",
        "PSM2_forcen_magnitude",
    ]

    ALL_FEATURES_NO_FORCEN = [
        "completion_time",
        "PSM1_average_speed_magnitude",
        "PSM2_average_speed_magnitude",
        "PSM1_average_acceleration_magnitude",
        "PSM2_average_acceleration_magnitude",
        "PSM1_average_jerk_magnitude",
        "PSM2_average_jerk_magnitude",
        "ATIForceSensor_average_force_magnitude",
        "PSM1_total_path_length",
        "PSM2_total_path_length",
        "PSM1_average_angular_speed_magnitude",
        "PSM2_average_angular_speed_magnitude",
        "PSM1_PSM2_speed_correlation",
        "PSM1_PSM2_speed_cross",
        "PSM1_PSM2_acceleration_cross",
        "PSM1_PSM2_jerk_cross",
        "PSM1_PSM2_acceleration_dispertion",
        "PSM1_PSM2_jerk_dispertion",
    ]

    # CHOSE FEATURE SUBSET 
    FEATURES = ALL_FEATURES_NO_FORCEN

    # Model
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)

    # Optimization
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    # Misc
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--outdir", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    if args.validation_mode == "split":
        # Current scheme (manual CSVs)
        train_ds = MetricsMLPDataset(args.train_csv, normalize=True, features=FEATURES)
        val_ds   = MetricsMLPDataset(args.val_csv, normalize=True,
                                     norm_mean=train_ds.feature_mean,
                                     norm_std=train_ds.feature_std,
                                     features=FEATURES)
        ckpt_path = os.path.join(args.outdir, "mlp_best.pt")
        run_training(train_ds, val_ds, args, device, ckpt_path)

    elif args.validation_mode == "louo":
        if args.all_csv is None:
            raise ValueError("--all_csv is required for LOUO mode")

        df = pd.read_csv(args.all_csv)
        # subjects = df["subject_id"].unique()
        # Split "Subject_Trial" into subject_id and trial_id
        df[["subject_id", "trial_id"]] = df["Subject_Trial"].str.split("_", expand=True)

        # Now subjects are accessible
        subjects = df["subject_id"].unique()

        fold_accs = []  
        all_true, all_pred = [], []
        for subj in subjects:
            print(f"\n=== LOUO fold: leaving out subject {subj} ===")
            train_df = df[df["subject_id"] != subj]
            val_df   = df[df["subject_id"] == subj]

            train_df.to_csv("tmp_train.csv", index=False)
            val_df.to_csv("tmp_val.csv", index=False)

            train_ds = MetricsMLPDataset("tmp_train.csv", normalize=True, features=FEATURES)
            val_ds   = MetricsMLPDataset("tmp_val.csv", normalize=True,
                                         norm_mean=train_ds.feature_mean,
                                         norm_std=train_ds.feature_std,
                                         features=FEATURES)

            ckpt_path = os.path.join(args.outdir, f"mlp_best_subj{subj}.pt")
            _, val_acc = run_training(train_ds, val_ds, args, device, ckpt_path)
            fold_accs.append((subj, val_acc))

            # Reload model and evaluate with return_preds=True
            model = metricsMLP(input_dim=train_ds.n_features,
                               hidden1=args.hidden1,
                               hidden2=args.hidden2,
                               num_classes=train_ds.n_classes).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            criterion = nn.CrossEntropyLoss()
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
            val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device, return_preds=True)

            # for confusion matrix 
            all_true.extend(targets)
            all_pred.extend(preds)

        print("\n=== LOUO Summary ===")
        for subj, acc in fold_accs:
            print(f"Subject {subj}: Accuracy = {acc:.4f}")
        print(f"Mean acc: {np.mean([acc for _, acc in fold_accs]):.4f} ± {np.std([acc for _, acc in fold_accs]):.4f}")

        labels = [0, 1, 2]
        class_names = [str(l) for l in labels]
        plot_confusion_matrix(all_true, all_pred, labels, class_names, "LOUO Confusion Matrix")

    elif args.validation_mode == "loso":
        if args.all_csv is None:
            raise ValueError("--all_csv is required for LOSO mode")

        df = pd.read_csv(args.all_csv)
        if "subject_id" not in df.columns or "trial_id" not in df.columns:
            df[["subject_id", "trial_id"]] = df["Subject_Trial"].str.split("_", expand=True)

        supertrials = df["trial_id"].unique()

        ## HARDCODED (supertrial that are less than 10)
        short_supertrials = ["T11", "T10", "T09", "T08", "T07"]

        fold_accs = []
        for trial in supertrials:
            
            ## skipping all short supertrials ( < 10 trials)
            if any(supertrial in trial for supertrial in short_supertrials):
                continue

            print(f"\n=== LOSO fold: leaving out supertrial {trial} ===")
            train_df = df[df["trial_id"] != trial]
            val_df = df[df["trial_id"] == trial]

            train_df.to_csv("tmp_train.csv", index=False)
            val_df.to_csv("tmp_val.csv", index=False)

            train_ds = MetricsMLPDataset("tmp_train.csv", normalize=True, features=FEATURES)
            val_ds = MetricsMLPDataset("tmp_val.csv", normalize=True,
                                       norm_mean=train_ds.feature_mean,
                                       norm_std=train_ds.feature_std,
                                       features=FEATURES)

            ckpt_path = os.path.join(args.outdir, f"mlp_best_supertrial{trial}.pt")
            _, val_acc = run_training(train_ds, val_ds, args, device, ckpt_path)
            fold_accs.append((trial, val_acc))

        print("\n=== LOSO Summary ===")
        for trial, acc in fold_accs:
            print(f"Supertrial {trial}: Accuracy = {acc:.4f}")
        print(f"Mean acc: {np.mean([acc for _, acc in fold_accs]):.4f} ± {np.std([acc for _, acc in fold_accs]):.4f}")


if __name__ == "__main__":
    main()