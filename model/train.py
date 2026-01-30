import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from dataset import MetricsMLPDataset
from network import metricsMLP
from utils import set_seed, plot_confusion_matrix, plot_loss_curves
from utils import evaluate, run_training, compute_permutation_importance, plot_permutation_importance
import feats

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

   

    # Choose feature subset
    # FEATURES = feats.ALL_FEATURESßß
    # FEATURES = feats.ALL_FEATURES_NO_FORCEN
    FEATURES = feats.KINEMATICS
    # FEATURES = feats.ALL_FEATURES
    # Example: mix kinematics + selected kinetics
    # FEATURES = feats.KINEMATICS + feats.KINETICS[8:12]
    # FEATURES = feats.KINETICS
    
    # Model
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=64)

    # Optimization
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop if val_loss doesn't improve for this many epochs (0 disables).")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum val_loss decrease to count as improvement.")

    # Misc
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--outdir", type=str, default="./checkpoints")
    parser.add_argument("--eval_ds", type=str, default="val")
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
        all_importances = []
        for fold_idx, subj in enumerate(subjects):
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
            if fold_idx == 5:
                _, val_acc, history = run_training(
                    train_ds, val_ds, args, device, ckpt_path, return_history=True
                )
                plot_path = os.path.join(args.outdir, "louo_first_fold_loss.png")
                plot_loss_curves(
                    history["train_loss"],
                    history["val_loss"],
                    plot_path,
                    title=f"LOUO First Fold Loss (subject {subj})",
                )
            else:
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

            # Compute permutation importance per fold for aggregation
            if args.eval_ds == "val":
                importance_scores = compute_permutation_importance(
                    model, val_ds, nn.CrossEntropyLoss(), device, FEATURES, n_repeats=20
                )
                all_importances.append(importance_scores)
            elif args.eval_ds == "train":
                importance_scores = compute_permutation_importance(
                    model, train_ds, nn.CrossEntropyLoss(), device, FEATURES, n_repeats=20
                )
                all_importances.append(importance_scores)

        print("\n=== LOUO Summary ===")
        for subj, acc in fold_accs:
            print(f"Subject {subj}: Accuracy = {acc:.4f}")
        print(f"Mean acc: {np.mean([acc for _, acc in fold_accs]):.4f} ± {np.std([acc for _, acc in fold_accs]):.4f}")

        labels = [0, 1, 2]
        class_names = [str(l) for l in labels]
        plot_confusion_matrix(all_true, all_pred, labels, class_names, "LOUO Confusion Matrix")
        print("\n=== Permutation Importance (LOUO aggregate model) ===")

        # Aggregate permutation importance across folds
        if all_importances:
            mean_importance_scores = {}
            for feature in FEATURES:
                vals = [imp.get(feature, 0) for imp in all_importances]
                mean_importance_scores[feature] = np.mean(vals)

            for feature, mean_val in mean_importance_scores.items():
                print(f"{feature}: Δacc = {mean_val:.6f}")
            plot_permutation_importance(importances=mean_importance_scores)

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
