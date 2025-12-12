import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import os
import sys

# === CONFIGURATION ===
CSV_PATH = sys.argv[1]
OUT_DIR = "correlation_curves_reduced_dataset_1"
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# Encode labels numerically (NOVICE=0, INTERMEDIATE=1, EXPERT=2)
label_encoder = LabelEncoder()
df["LabelNum"] = label_encoder.fit_transform(df["LABEL"].astype(str).str.upper())

# Select numeric feature columns
exclude_cols = {"Subject_Trial", "LABEL", "LabelNum"}
feature_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]

# === STORE CORRELATIONS ===
correlations = {}

# === FIT AND PLOT EACH FEATURE ===
for feature in feature_cols:
    x = df["LabelNum"].values.reshape(-1, 1)
    y = df[feature].values

    # Fit simple linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Compute correlation coefficient
    r, _ = pearsonr(df["LabelNum"], y)
    correlations[feature] = r

    # === Plot individual regression ===
    plt.figure(figsize=(6, 4))
    plt.scatter(df["LabelNum"], y, alpha=0.7, label="Data points")
    plt.plot(df["LabelNum"], y_pred, color="red", linewidth=2, label=f"Linear fit (r={r:.2f})")

    plt.title(f"{feature} vs Skill Level")
    plt.xlabel("Skill level (0=Novice, 1=Intermediate, 2=Expert)")
    plt.ylabel(feature)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"linear_fit_{feature}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"Saved {len(feature_cols)} individual plots to {OUT_DIR}")

# === CREATE CORRELATION SUMMARY PLOT ===
sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
features_sorted = list(sorted_corr.keys())
r_values = [sorted_corr[f] for f in features_sorted]

plt.figure(figsize=(10, max(5, len(features_sorted) * 0.3)))
bars = plt.barh(features_sorted, r_values, color=["#2a9d8f" if r >= 0 else "#e76f51" for r in r_values])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Skill Level")
plt.xlabel("Pearson correlation coefficient (r)")
plt.ylabel("Metric")
plt.tight_layout()

# Save correlation summary plot
summary_path = os.path.join(OUT_DIR, "feature_correlations_summary.png")
plt.savefig(summary_path, dpi=300)
plt.close()

# === ALSO SAVE AS CSV ===
corr_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Correlation"])
corr_df.to_csv(os.path.join(OUT_DIR, "feature_correlations.csv"), index=False)

print(f"✅ Saved correlation summary plot and CSV to {OUT_DIR}")

# === ALSO SAVE AS CSV AND PRINT TABLE ===
corr_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Correlation"])

# Sort by absolute correlation strength
corr_df = corr_df.reindex(corr_df["Correlation"].abs().sort_values(ascending=False).index)

# Print formatted table
print("\n=== Correlation Coefficients (sorted by |r|) ===")
print(corr_df.to_string(index=False, justify="left", col_space=20, formatters={
    "Correlation": lambda x: f"{x:.4f}"
}))

# Save to CSV
corr_csv_path = os.path.join(OUT_DIR, "feature_correlations.csv")
corr_df.to_csv(corr_csv_path, index=False)

print(f"\n✅ Saved correlation summary plot and CSV to {OUT_DIR}")