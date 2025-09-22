import os
import pandas as pd

def split_data_vanilla(input_csv, split_ratio=.8):
    """
    Reads the CSV at input_csv, shuffles rows, splits into train/val sets, saves them, and returns DataFrames.
    """
    # Read CSV
    df = pd.read_csv(input_csv)
    # Shuffle with fixed seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Compute split index
    split_idx = int(len(df_shuffled) * split_ratio)
    train_df = df_shuffled.iloc[:split_idx].reset_index(drop=True)
    val_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)
    # Determine output directory
    out_dir = os.path.dirname(input_csv)
    train_csv = os.path.join(out_dir, "train_force.csv")
    val_csv = os.path.join(out_dir, "val_force.csv")
    # Save CSVs
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    # Return DataFrames
    return train_df, val_df

split_data_vanilla("../metrics/results_ml_force.csv")