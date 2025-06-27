import pandas as pd
import os

def align_and_downsample(
    parquet_files,
    baseline_file,
    target_frequency="20.833ms",
    output_dir="aligned_data"
):
    """
    Aligns and downsamples Parquet files to a baseline file's timestamps, preserves original timestamps,
    and calculates alignment error (residual) for each file.

    Args:
        parquet_files (list): List of Parquet file paths.
        baseline_file (str): Path to the baseline Parquet file.
        target_frequency (str): Target frequency for downsampling (e.g., '20.833ms').
        output_dir (str): Directory to save the aligned and downsampled Parquet files.

    Returns:
        dict: Alignment errors (mean residuals in seconds) for each file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load baseline file and create global index
    baseline_df = pd.read_parquet(baseline_file)
    baseline_df["timestamp"] = baseline_df.iloc[:, 1] + baseline_df.iloc[:, 2] / 1e9
    baseline_df["datetime"] = pd.to_datetime(baseline_df["timestamp"], unit="s")
    baseline_df = baseline_df.set_index("datetime").sort_index()
    baseline_index = pd.date_range(
        start=baseline_df.index.min(),
        end=baseline_df.index.max(),
        freq=target_frequency
    )

    alignment_errors = {}

    for file in parquet_files:
        if file == baseline_file:
            continue  # Skip baseline file itself

        df = pd.read_parquet(file)
        # Preserve original timestamps as columns
        df["orig_header_stamp_sec"] = df.iloc[:, 1]
        df["orig_header_stamp_nsec"] = df.iloc[:, 2]
        df["orig_datetime"] = pd.to_datetime(df["orig_header_stamp_sec"] + df["orig_header_stamp_nsec"] / 1e9, unit="s")

        # Set index for alignment
        df["timestamp"] = df.iloc[:, 1] + df.iloc[:, 2] / 1e9
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.infer_objects(copy=False)
        numeric_columns = df.select_dtypes(include=["number"]).columns
        df = df[numeric_columns.union(["orig_header_stamp_sec", "orig_header_stamp_nsec", "orig_datetime"])]

        # Align and downsample to baseline index trimmed to the file's time range
        valid_index = baseline_index[(baseline_index >= df.index.min()) & (baseline_index <= df.index.max())]
        aligned_df = df.reindex(valid_index).interpolate(method="linear")

        # Calculate alignment error (residual) for original timestamps to nearest baseline timestamp
        # Ensure orig_times is datetime64[ns] and drop NaNs
        orig_times = pd.to_datetime(df["orig_datetime"], errors='coerce')
        orig_times = orig_times[orig_times.notnull()]

        # Ensure baseline_times is a DatetimeIndex
        baseline_times = pd.DatetimeIndex(baseline_index)

        def nearest_residual(ts):
            if pd.isnull(ts):
                return None
            # Find nearest baseline timestamp
            idx = baseline_times.get_indexer([ts], method="nearest")[0]
            nearest = baseline_times[idx]
            return abs((ts - nearest).total_seconds())

        residuals = orig_times.apply(nearest_residual).dropna()
        mean_residual = residuals.mean()
        alignment_errors[file] = mean_residual

        print(f"Mean alignment error for {os.path.basename(file)}: {mean_residual:.6f} seconds")

        # Save aligned DataFrame with preserved original timestamps
        output_file = os.path.join(output_dir, os.path.basename(file))
        aligned_df.to_parquet(output_file)
        print(f"Saved aligned and downsampled file: {output_file}")

    return alignment_errors

# --- Main script ---

path = "/Volumes/drakes_ssd_500gb/skill_assessment/data/S01/parquet/T01/"
all_entries = os.listdir(path)

# Filter out files containing "console" or "select"
parquet_files = [
    os.path.join(path, f)
    for f in all_entries
    if os.path.isfile(os.path.join(path, f))
    and ".parquet" in f
    and "console" not in f
    and "select" not in f
    and f != ".DS_Store"
]

print(f"Files to process: {len(parquet_files)}")
print(parquet_files)

# Choose a baseline file (for example, the first one)
baseline_file = parquet_files[0]

alignment_errors = align_and_downsample(
    parquet_files,
    baseline_file=baseline_file,
    target_frequency="20.833ms",
    output_dir="aligned_data"
)

# Print alignment errors
for file, error in alignment_errors.items():
    print(f"Mean alignment error for {os.path.basename(file)}: {error:.6f} seconds")