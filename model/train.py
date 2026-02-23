import argparse
import os
import sys

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
from utils import collect_predictions, compute_fold_metrics, aggregate_louo_metrics
from utils import plot_multiclass_roc, save_roc_npz, save_metrics_json, save_pooled_predictions
from utils import build_louo_metrics_table, save_metrics_table_csv
from utils import run_louo_seed_sweep, aggregate_seed_sweep_metrics
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

   

    # Feature/metric metadata for reporting and experiment tracking
    parser.add_argument("--feature_set", type=str, default="kinetics",
                        choices=["all", "all_no_forcen", "kinematics", "kinetics", "force", "anova","perm", "perm8"],
                        help="Predefined feature set to use.")
    parser.add_argument("--custom_features", type=str, default=None,
                        help="Optional comma-separated feature names. Overrides --feature_set.")
    parser.add_argument("--feature_set_name", type=str, default=None,
                        help="Optional label for exported tables when using --custom_features.")
    parser.add_argument("--metric_family", type=str, default="classification",
                        help="Metric family label for exports (e.g., classification, kinetics, kinematics).")
    
    # Model
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=64)

    # Optimization
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop if val_loss doesn't improve for this many epochs (0 disables).")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum val_loss decrease to count as improvement.")

    # Misc
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--outdir", type=str, default="./checkpoints")
    parser.add_argument("--eval_ds", type=str, default="val")
    parser.add_argument("--bootstrap_folds", type=int, default=1000,
                        help="Bootstrap iterations per fold for sens/spec CIs.")
    parser.add_argument("--bootstrap_pooled", type=int, default=2000,
                        help="Bootstrap iterations for pooled sens/spec CIs.")
    parser.add_argument("--disable_bootstrap", action="store_true",
                        help="Disable bootstrap CIs and report point estimates only.")
    parser.add_argument("--metrics_outdir", type=str, default=None,
                        help="Optional override for metrics output directory.")
    parser.add_argument("--metrics_table_path", type=str, default=None,
                        help="Optional explicit CSV path for the long-format metrics table.")
    parser.add_argument("--append_metrics_table", action="store_true",
                        help="Append current run rows to existing metrics table instead of overwriting.")
    parser.add_argument("--seed_sweep_runs", type=int, default=1,
                        help="If >1, run LOUO multiple times with random seeds and summarize results.")
    parser.add_argument("--seed_sweep_rng_seed", type=int, default=42,
                        help="RNG seed used to sample run seeds for --seed_sweep_runs.")
    parser.add_argument("--seed_sweep_seed_low", type=int, default=1,
                        help="Lower bound (inclusive) for sampled run seeds.")
    parser.add_argument("--seed_sweep_seed_high", type=int, default=1_000_000,
                        help="Upper bound (inclusive) for sampled run seeds.")
    parser.add_argument("--seed_sweep_out_csv", type=str, default=None,
                        help="Optional CSV path for seed sweep summary output.")
    args = parser.parse_args()

    if args.seed_sweep_runs > 1:
        if args.validation_mode != "louo":
            raise ValueError("--seed_sweep_runs > 1 currently supports only --validation_mode louo.")

        def _strip_flags(argv, flags_with_values, bool_flags):
            stripped = []
            i = 0
            while i < len(argv):
                tok = argv[i]
                if tok in flags_with_values:
                    i += 2
                    continue
                if tok in bool_flags:
                    i += 1
                    continue
                stripped.append(tok)
                i += 1
            return stripped

        flags_with_values = {
            "--seed",
            "--metrics_outdir",
            "--seed_sweep_runs",
            "--seed_sweep_rng_seed",
            "--seed_sweep_seed_low",
            "--seed_sweep_seed_high",
            "--seed_sweep_out_csv",
        }
        bool_flags = set()
        forwarded_cli = _strip_flags(sys.argv[1:], flags_with_values, bool_flags)
        script_path = os.path.abspath(__file__)
        base_cmd = [sys.executable, script_path] + forwarded_cli

        metrics_root = args.metrics_outdir or os.path.join(args.outdir, "metrics", "seed_sweep")
        summary_df = run_louo_seed_sweep(
            n_runs=args.seed_sweep_runs,
            base_cmd=base_cmd,
            metrics_root=metrics_root,
            rng_seed=args.seed_sweep_rng_seed,
            seed_low=args.seed_sweep_seed_low,
            seed_high=args.seed_sweep_seed_high,
            workdir=os.path.dirname(script_path),
        )
        out_csv = args.seed_sweep_out_csv or os.path.join(metrics_root, "seed_sweep_summary.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        summary_df.to_csv(out_csv, index=False)
        seed_table_df = aggregate_seed_sweep_metrics(summary_df["metrics_dir"].tolist())
        if not seed_table_df.empty:
            row_feature_set = str(seed_table_df.iloc[0]["feature_set"])
            row_metric_family = str(seed_table_df.iloc[0]["metric_family"])
            acc_row = {
                "feature_set": row_feature_set,
                "metric_family": row_metric_family,
                "metric_name": "accuracy",
                "class_name": "overall",
                "eval_level": "seed_sweep",
                "estimate": float(summary_df["accuracy"].mean()),
                "spread_type": "std",
                "spread_value": float(summary_df["accuracy"].std(ddof=0)),
                "lower": np.nan,
                "upper": np.nan,
                "n_folds": np.nan,
                "n_samples": np.nan,
                "n_runs": int(summary_df.shape[0]),
            }
            seed_table_df = pd.concat([seed_table_df, pd.DataFrame([acc_row])], ignore_index=True)
            seed_table_path = os.path.join(metrics_root, "louo_metrics_table_seed_sweep.csv")
            seed_table_df.to_csv(seed_table_path, index=False)
        else:
            seed_table_path = None

        print("\n=== LOUO Seed Sweep Summary ===")
        print(summary_df.to_string(index=False))
        print(
            f"\nMean accuracy: {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std(ddof=0):.4f}\n"
            f"Mean auc_macro: {summary_df['auc_macro'].mean():.4f} ± {summary_df['auc_macro'].std(ddof=0):.4f}\n"
            f"Mean auc_weighted: {summary_df['auc_weighted'].mean():.4f} ± {summary_df['auc_weighted'].std(ddof=0):.4f}\n"
            f"Saved summary CSV: {out_csv}"
        )
        if seed_table_path:
            print(f"Saved seed-sweep metrics table: {seed_table_path}")
        return

    feature_set_map = {
        "all": feats.ALL_FEATURES,
        "all_no_forcen": feats.ALL_FEATURES_NO_FORCEN,
        "kinematics": feats.KINEMATICS,
        "kinetics": feats.KINETICS,
        "force": feats.FORCE_FEATURES,
        "anova": feats.ANOVA_FEATURES_16,
        "perm": feats.PERM_IMPORTANCE_FEATURES_16,
        "perm8": feats.PERM_IMPORTANCE_FEATURES_8
    }

    if args.custom_features:
        FEATURES = [f.strip() for f in args.custom_features.split(",") if f.strip()]
        if not FEATURES:
            raise ValueError("--custom_features was provided but no valid feature names were parsed.")
        feature_set_name = args.feature_set_name or "custom"
    else:
        FEATURES = feature_set_map[args.feature_set]
        feature_set_name = args.feature_set

    print("FEATURES FEATURES FEATURES")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    n_boot_folds = 0 if args.disable_bootstrap else args.bootstrap_folds
    n_boot_pooled = 0 if args.disable_bootstrap else args.bootstrap_pooled


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
        all_score = []
        all_importances = []
        fold_metrics = []
        class_names = None
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
            if class_names is None:
                inv_label_mapping = {v: k for k, v in train_ds.label_mapping.items()}
                class_names = [inv_label_mapping[i].title() for i in range(train_ds.n_classes)]

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

            # Collect probabilities for ROC/AUC
            y_true, y_pred, y_score = collect_predictions(model, val_loader, device)
            all_score.append(y_score)

            # Fold-level metrics
            metrics = compute_fold_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                class_names=class_names,
                n_boot=n_boot_folds,
                alpha=0.05,
            )
            fold_metrics.append(metrics)

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

        if all_score:
            n_classes = all_score[0].shape[1]
        else:
            n_classes = 0
        labels = list(range(n_classes))
        if class_names is None:
            class_names = [str(l) for l in labels]
        if n_classes:
            plot_confusion_matrix(all_true, all_pred, labels, class_names, "LOUO Confusion Matrix")

        # Aggregate ROC/AUC + sensitivity/specificity across folds
        if all_score:
            pooled_true = np.array(all_true)
            pooled_pred = np.array(all_pred)
            pooled_score = np.concatenate(all_score, axis=0)
            metrics_dir = args.metrics_outdir or os.path.join(args.outdir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            agg = aggregate_louo_metrics(
                fold_metrics=fold_metrics,
                pooled_true=pooled_true,
                pooled_pred=pooled_pred,
                pooled_score=pooled_score,
                class_names=class_names,
                n_boot_pooled=n_boot_pooled,
                alpha=0.05,
            )
            if agg:
                fs = agg["fold_summary"]
                print("\n=== LOUO ROC/AUC Summary (fold mean ± std) ===")
                print(f"AUC macro:    {fs['auc_macro_mean']:.4f} ± {fs['auc_macro_std']:.4f}")
                print(f"AUC weighted: {fs['auc_weighted_mean']:.4f} ± {fs['auc_weighted_std']:.4f}")
                print("\n=== LOUO Sens/Spec (fold mean ± std) ===")
                for i, name in enumerate(fs["class_names"]):
                    print(
                        f"Class {name}: "
                        f"Sens {fs['sensitivity_mean'][i]:.4f} ± {fs['sensitivity_std'][i]:.4f} | "
                        f"Spec {fs['specificity_mean'][i]:.4f} ± {fs['specificity_std'][i]:.4f}"
                    )

                pooled = agg["pooled"]
                print("\n=== LOUO Pooled ROC/AUC ===")
                print(f"AUC macro:    {pooled['auc_macro']:.4f}")
                print(f"AUC weighted: {pooled['auc_weighted']:.4f}")
                if args.disable_bootstrap:
                    print("\n=== LOUO Pooled Sens/Spec (point estimates; bootstrap disabled) ===")
                else:
                    print("\n=== LOUO Pooled Sens/Spec (95% CI) ===")
                sens = pooled["sens_spec"]["sensitivity"]
                sens_ci = pooled["sens_spec"]["sensitivity_ci"]
                spec = pooled["sens_spec"]["specificity"]
                spec_ci = pooled["sens_spec"]["specificity_ci"]
                for i, name in enumerate(pooled["class_names"]):
                    if args.disable_bootstrap:
                        print(f"{name}: Sens {sens[i]:.4f} | Spec {spec[i]:.4f}")
                    else:
                        print(
                            f"{name}: "
                            f"Sens {sens[i]:.4f} [{sens_ci[i][0]:.4f}, {sens_ci[i][1]:.4f}] | "
                            f"Spec {spec[i]:.4f} [{spec_ci[i][0]:.4f}, {spec_ci[i][1]:.4f}]"
                        )

                # Save metrics and pooled ROC to disk
                save_metrics_json(agg, os.path.join(metrics_dir, "louo_metrics.json"))
                save_roc_npz(pooled["roc"], os.path.join(metrics_dir, "louo_roc_pooled.npz"))
                plot_multiclass_roc(
                    pooled["roc"],
                    os.path.join(metrics_dir, "louo_roc_pooled.png"),
                    title="LOUO Pooled ROC",
                )
                save_pooled_predictions(
                    os.path.join(metrics_dir, "louo_pooled_predictions.npz"),
                    pooled_true,
                    pooled_pred,
                    pooled_score,
                )
                summary_table = build_louo_metrics_table(
                    agg=agg,
                    feature_set=feature_set_name,
                    metric_family=args.metric_family,
                    n_folds=len(fold_metrics),
                    n_samples=int(pooled_true.shape[0]),
                )
                metrics_table_path = args.metrics_table_path or os.path.join(metrics_dir, "louo_metrics_table.csv")
                save_metrics_table_csv(
                    summary_table,
                    metrics_table_path,
                    append=args.append_metrics_table,
                )
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
            metrics_dir = args.metrics_outdir or os.path.join(args.outdir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            imp_df = pd.DataFrame(
                {"feature": list(mean_importance_scores.keys()),
                 "delta_acc": list(mean_importance_scores.values())}
            )
            imp_df.to_csv(os.path.join(metrics_dir, "louo_permutation_importance.csv"), index=False)

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
