import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import json
import pandas as pd
import os
import subprocess
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
    best_val_acc = 0.0
    best_epoch = -1
    no_improve_epochs = 0
    early_stop_patience = getattr(args, "early_stop_patience", 0)
    early_stop_min_delta = getattr(args, "early_stop_min_delta", 0.0)
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

        if val_loss < (best_val_loss - early_stop_min_delta):
        # if 1:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save({"epoch": epoch,"model_state": model.state_dict(),"optimizer_state": optimizer.state_dict(),"meta": {"n_features": train_ds.n_features,
                         "n_classes": train_ds.n_classes},"args": vars(args),}, ckpt_path)
        else:
            no_improve_epochs += 1

        if early_stop_patience and no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch:03d} "
                f"(no improvement in val_loss for {early_stop_patience} epochs)."
            )
            break

    print(f"Best epoch: {best_epoch} | best_val_loss: {best_val_loss:.4f}\nSaved: {ckpt_path}")
    if return_history:
        history = {"train_loss": train_losses, "val_loss": val_losses}
        return best_val_loss, best_val_acc, history
    return best_val_loss, best_val_acc



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


# ---------------------------
# Evaluation Metrics
# ---------------------------

def compute_roc_curves_multiclass(y_true, y_score, class_names=None):
    """
    y_true: array-like shape (n_samples,)
    y_score: array-like shape (n_samples, n_classes) - probabilities or logits
    Returns per-class ROC data and AUCs.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_classes = y_score.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    roc_data = {}
    for c in range(n_classes):
        y_true_bin = (y_true == c).astype(int)
        # If only one class present in y_true_bin, ROC is undefined
        if y_true_bin.max() == y_true_bin.min():
            roc_data[class_names[c]] = {
                "fpr": None,
                "tpr": None,
                "thresholds": None,
                "auc": np.nan,
            }
            continue
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_score[:, c])
        roc_data[class_names[c]] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc(fpr, tpr),
        }
    return roc_data


def compute_multiclass_auc(y_true, y_score, average="macro", multi_class="ovr"):
    """
    Single summary AUC for multiclass.
    average: "macro" or "weighted"
    multi_class: "ovr" or "ovo"
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_classes = y_score.shape[1]
    labels = list(range(n_classes))
    return roc_auc_score(
        y_true,
        y_score,
        average=average,
        multi_class=multi_class,
        labels=labels,
    )


def _sens_spec_from_cm(cm):
    # cm shape (n_classes, n_classes)
    n_classes = cm.shape[0]
    sens = np.zeros(n_classes, dtype=float)
    spec = np.zeros(n_classes, dtype=float)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sens[i] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec[i] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec


def bootstrap_sens_spec_ci(y_true, y_pred, n_classes, n_boot=1000, alpha=0.05, seed=42):
    """
    Returns per-class sensitivity/specificity with bootstrap CIs.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]

    base_cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    base_sens, base_spec = _sens_spec_from_cm(base_cm)

    if n_boot is None or n_boot <= 0:
        nan_ci = np.full((n_classes, 2), np.nan, dtype=float)
        return {
            "sensitivity": base_sens,
            "sensitivity_ci": nan_ci,
            "specificity": base_spec,
            "specificity_ci": nan_ci.copy(),
        }

    sens_samples = []
    spec_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cm = confusion_matrix(y_true[idx], y_pred[idx], labels=list(range(n_classes)))
        s, sp = _sens_spec_from_cm(cm)
        sens_samples.append(s)
        spec_samples.append(sp)

    sens_samples = np.vstack(sens_samples)
    spec_samples = np.vstack(spec_samples)
    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0

    sens_ci = np.nanquantile(sens_samples, [lo, hi], axis=0).T
    spec_ci = np.nanquantile(spec_samples, [lo, hi], axis=0).T

    return {
        "sensitivity": base_sens,
        "sensitivity_ci": sens_ci,
        "specificity": base_spec,
        "specificity_ci": spec_ci,
    }


def collect_predictions(model, loader, device):
    """
    Returns y_true, y_pred, y_score (softmax probabilities) as numpy arrays.
    """
    model.eval()
    all_true, all_pred, all_score = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_true.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_score.append(probs.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_score = np.concatenate(all_score)
    return y_true, y_pred, y_score


def compute_fold_metrics(y_true, y_pred, y_score, class_names=None, n_boot=1000, alpha=0.05):
    """
    Computes ROC curves, multiclass AUC, and sensitivity/specificity with CIs for one fold.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    n_classes = y_score.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    roc_data = compute_roc_curves_multiclass(y_true, y_score, class_names=class_names)
    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        auc_macro = np.nan
        auc_weighted = np.nan
    else:
        auc_macro = compute_multiclass_auc(y_true, y_score, average="macro", multi_class="ovr")
        auc_weighted = compute_multiclass_auc(y_true, y_score, average="weighted", multi_class="ovr")
    sens_spec = bootstrap_sens_spec_ci(
        y_true, y_pred, n_classes=n_classes, n_boot=n_boot, alpha=alpha
    )

    return {
        "roc": roc_data,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "sens_spec": sens_spec,
    }


def aggregate_louo_metrics(
    fold_metrics,
    pooled_true,
    pooled_pred,
    pooled_score,
    class_names=None,
    n_boot_pooled=2000,
    alpha=0.05,
):
    """
    Aggregates fold metrics with mean/std and pooled metrics over all folds.
    """
    if not fold_metrics:
        return {}

    n_classes = pooled_score.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    auc_macros = np.array([m["auc_macro"] for m in fold_metrics], dtype=float)
    auc_weighteds = np.array([m["auc_weighted"] for m in fold_metrics], dtype=float)

    # Per-class sens/spec across folds
    sens = np.vstack([m["sens_spec"]["sensitivity"] for m in fold_metrics])
    spec = np.vstack([m["sens_spec"]["specificity"] for m in fold_metrics])

    pooled_auc_macro = compute_multiclass_auc(pooled_true, pooled_score, average="macro", multi_class="ovr")
    pooled_auc_weighted = compute_multiclass_auc(pooled_true, pooled_score, average="weighted", multi_class="ovr")
    pooled_sens_spec = bootstrap_sens_spec_ci(
        pooled_true, pooled_pred, n_classes=n_classes, n_boot=n_boot_pooled, alpha=alpha
    )
    pooled_roc = compute_roc_curves_multiclass(pooled_true, pooled_score, class_names=class_names)

    def _safe_nanmean(x):
        return float(np.nan) if np.all(np.isnan(x)) else float(np.nanmean(x))

    def _safe_nanstd(x):
        return float(np.nan) if np.all(np.isnan(x)) else float(np.nanstd(x))

    return {
        "fold_summary": {
            "auc_macro_mean": _safe_nanmean(auc_macros),
            "auc_macro_std": _safe_nanstd(auc_macros),
            "auc_weighted_mean": _safe_nanmean(auc_weighteds),
            "auc_weighted_std": _safe_nanstd(auc_weighteds),
            "sensitivity_mean": np.nanmean(sens, axis=0),
            "sensitivity_std": np.nanstd(sens, axis=0),
            "specificity_mean": np.nanmean(spec, axis=0),
            "specificity_std": np.nanstd(spec, axis=0),
            "class_names": class_names,
        },
        "pooled": {
            "auc_macro": pooled_auc_macro,
            "auc_weighted": pooled_auc_weighted,
            "sens_spec": pooled_sens_spec,
            "roc": pooled_roc,
            "class_names": class_names,
        },
    }


def plot_multiclass_roc(roc_data, outpath, title="ROC Curves (Multiclass)"):
    plt.figure(figsize=(7, 6))
    for cls_name, data in roc_data.items():
        if data["fpr"] is None or data["tpr"] is None:
            continue
        plt.plot(data["fpr"], data["tpr"], label=f"{cls_name} (AUC {data['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_npz(roc_data, outpath):
    arrays = {}
    for cls_name, data in roc_data.items():
        key = str(cls_name)
        arrays[f"fpr_{key}"] = np.asarray(data["fpr"])
        arrays[f"tpr_{key}"] = np.asarray(data["tpr"])
        arrays[f"thresholds_{key}"] = np.asarray(data["thresholds"])
        arrays[f"auc_{key}"] = np.asarray([data["auc"]])
    np.savez(outpath, **arrays)


def save_pooled_predictions(outpath, y_true, y_pred, y_score):
    np.savez(
        outpath,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
    )


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_metrics_json(metrics, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, indent=2)


def build_louo_metrics_table(
    agg,
    feature_set,
    metric_family,
    n_folds=None,
    n_samples=None,
):
    """
    Build a long-format table for LOUO metrics with fold summaries and pooled results.
    """
    rows = []
    fold_summary = agg.get("fold_summary", {})
    pooled = agg.get("pooled", {})
    class_names = fold_summary.get("class_names", pooled.get("class_names", []))

    def _add_row(
        metric_name,
        class_name,
        eval_level,
        estimate,
        spread_type=None,
        spread_value=np.nan,
        lower=np.nan,
        upper=np.nan,
    ):
        rows.append(
            {
                "feature_set": feature_set,
                "metric_family": metric_family,
                "metric_name": metric_name,
                "class_name": class_name,
                "eval_level": eval_level,
                "estimate": float(estimate) if estimate is not None else np.nan,
                "spread_type": spread_type,
                "spread_value": float(spread_value) if spread_value is not None else np.nan,
                "lower": float(lower) if lower is not None else np.nan,
                "upper": float(upper) if upper is not None else np.nan,
                "n_folds": n_folds,
                "n_samples": n_samples,
            }
        )

    # Fold-level mean/std summaries
    _add_row(
        metric_name="auc_macro",
        class_name="overall",
        eval_level="fold",
        estimate=fold_summary.get("auc_macro_mean", np.nan),
        spread_type="std",
        spread_value=fold_summary.get("auc_macro_std", np.nan),
    )
    _add_row(
        metric_name="auc_weighted",
        class_name="overall",
        eval_level="fold",
        estimate=fold_summary.get("auc_weighted_mean", np.nan),
        spread_type="std",
        spread_value=fold_summary.get("auc_weighted_std", np.nan),
    )

    sens_mean = fold_summary.get("sensitivity_mean", [])
    sens_std = fold_summary.get("sensitivity_std", [])
    spec_mean = fold_summary.get("specificity_mean", [])
    spec_std = fold_summary.get("specificity_std", [])
    for i, class_name in enumerate(class_names):
        _add_row(
            metric_name="sensitivity",
            class_name=class_name,
            eval_level="fold",
            estimate=sens_mean[i],
            spread_type="std",
            spread_value=sens_std[i],
        )
        _add_row(
            metric_name="specificity",
            class_name=class_name,
            eval_level="fold",
            estimate=spec_mean[i],
            spread_type="std",
            spread_value=spec_std[i],
        )

    # Pooled summaries
    _add_row(
        metric_name="auc_macro",
        class_name="overall",
        eval_level="pooled",
        estimate=pooled.get("auc_macro", np.nan),
    )
    _add_row(
        metric_name="auc_weighted",
        class_name="overall",
        eval_level="pooled",
        estimate=pooled.get("auc_weighted", np.nan),
    )

    pooled_sens_spec = pooled.get("sens_spec", {})
    pooled_sens = pooled_sens_spec.get("sensitivity", [])
    pooled_spec = pooled_sens_spec.get("specificity", [])
    pooled_sens_ci = pooled_sens_spec.get("sensitivity_ci", [])
    pooled_spec_ci = pooled_sens_spec.get("specificity_ci", [])
    for i, class_name in enumerate(class_names):
        _add_row(
            metric_name="sensitivity",
            class_name=class_name,
            eval_level="pooled",
            estimate=pooled_sens[i],
            spread_type="ci95",
            lower=pooled_sens_ci[i][0],
            upper=pooled_sens_ci[i][1],
        )
        _add_row(
            metric_name="specificity",
            class_name=class_name,
            eval_level="pooled",
            estimate=pooled_spec[i],
            spread_type="ci95",
            lower=pooled_spec_ci[i][0],
            upper=pooled_spec_ci[i][1],
        )

    return pd.DataFrame(rows)


def save_metrics_table_csv(df, outpath, append=False):
    if append and os.path.exists(outpath):
        existing = pd.read_csv(outpath)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(outpath, index=False)
        return
    df.to_csv(outpath, index=False)


def run_louo_seed_sweep(
    n_runs,
    base_cmd,
    metrics_root,
    rng_seed=42,
    seed_low=1,
    seed_high=1_000_000,
    workdir=None,
):
    """
    Run LOUO training/eval N times with random seeds and return per-seed metrics.

    Parameters
    ----------
    n_runs : int
        Number of runs.
    base_cmd : list[str]
        Command list excluding --seed and --metrics_outdir.
        Example:
        [
            "python3", "train.py",
            "--all_csv", "../metrics/results_ml.csv",
            "--validation_mode", "louo",
            "--feature_set", "perm"
        ]
    metrics_root : str
        Root directory where per-seed output folders are written.
    rng_seed : int
        RNG seed used to sample unique run seeds.
    seed_low, seed_high : int
        Inclusive seed range to sample from.
    workdir : str or None
        Working directory for subprocess runs (e.g., "model").
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")
    if seed_high < seed_low:
        raise ValueError("seed_high must be >= seed_low.")

    seed_space = seed_high - seed_low + 1
    if n_runs > seed_space:
        raise ValueError(
            f"Cannot sample {n_runs} unique seeds from range [{seed_low}, {seed_high}]."
        )

    rng = np.random.default_rng(rng_seed)
    seeds = rng.choice(np.arange(seed_low, seed_high + 1), size=n_runs, replace=False)

    rows = []
    os.makedirs(metrics_root, exist_ok=True)
    for seed in seeds:
        run_metrics_dir = os.path.join(metrics_root, f"seed_{int(seed)}")
        cmd = list(base_cmd) + ["--seed", str(int(seed)), "--metrics_outdir", run_metrics_dir]
        subprocess.run(cmd, check=True, cwd=workdir)

        metrics_path = os.path.join(run_metrics_dir, "louo_metrics.json")
        preds_path = os.path.join(run_metrics_dir, "louo_pooled_predictions.npz")

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        pooled = metrics.get("pooled", {})

        pred_data = np.load(preds_path)
        y_true = pred_data["y_true"]
        y_pred = pred_data["y_pred"]
        accuracy = float(np.mean(y_true == y_pred))

        rows.append(
            {
                "seed": int(seed),
                "accuracy": accuracy,
                "auc_macro": float(pooled.get("auc_macro", np.nan)),
                "auc_weighted": float(pooled.get("auc_weighted", np.nan)),
                "metrics_dir": run_metrics_dir,
            }
        )

    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def aggregate_seed_sweep_metrics(seed_metrics_dirs):
    """
    Aggregate pooled LOUO metrics across seed runs as mean ± std.

    Parameters
    ----------
    seed_metrics_dirs : list[str]
        Per-seed metrics directories containing louo_metrics_table.csv.
    """
    if not seed_metrics_dirs:
        return pd.DataFrame()

    pooled_tables = []
    for metrics_dir in seed_metrics_dirs:
        table_path = os.path.join(metrics_dir, "louo_metrics_table.csv")
        if not os.path.exists(table_path):
            continue
        df = pd.read_csv(table_path)
        pooled_df = df[df["eval_level"] == "pooled"].copy()
        if pooled_df.empty:
            continue
        pooled_tables.append(pooled_df)

    if not pooled_tables:
        return pd.DataFrame()

    all_pooled = pd.concat(pooled_tables, ignore_index=True)
    grouped = (
        all_pooled
        .groupby(["feature_set", "metric_family", "metric_name", "class_name"], dropna=False)["estimate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    rows = []
    for _, r in grouped.iterrows():
        rows.append(
            {
                "feature_set": r["feature_set"],
                "metric_family": r["metric_family"],
                "metric_name": r["metric_name"],
                "class_name": r["class_name"],
                "eval_level": "seed_sweep",
                "estimate": float(r["mean"]),
                "spread_type": "std",
                "spread_value": float(r["std"]),
                "lower": np.nan,
                "upper": np.nan,
                "n_folds": np.nan,
                "n_samples": np.nan,
                "n_runs": int(r["count"]),
            }
        )

    return pd.DataFrame(rows)
