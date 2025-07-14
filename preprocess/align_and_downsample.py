import pandas as pd
import numpy as np
import os
from preprocess import find_subject_files
from tqdm import tqdm
import shutil

def align_and_downsample(
    parquet_files,
    baseline_file,
    target_frequency="2.6455ms",
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
    # Correct timestamp using explicit column names
    baseline_df["timestamp"] = baseline_df["header_stamp_sec"] + baseline_df["header_stamp_nsec"] / 1e9
    baseline_df["datetime"] = pd.to_datetime(baseline_df["timestamp"], unit="s")
    baseline_df = baseline_df.set_index("datetime").sort_index()
    baseline_index = pd.date_range(
        start=baseline_df.index.min(),
        end=baseline_df.index.max(),
        freq=target_frequency
    )

    # Upsample by adding midpoints between each timestamp
    midpoints = baseline_index[:-1] + (baseline_index[1:] - baseline_index[:-1]) / 2
    baseline_index = baseline_index.union(midpoints).sort_values()

    alignment_errors = {}

    for file in parquet_files:
        if "follow_mode" in os.path.basename(file):
            print(f"Skipping file containing 'follow_mode': {file}")
            continue
        if "jaw" in os.path.basename(file):
            print(f"Skipping file containing 'jaw': {file}")
            continue
        if "PSM3" in os.path.basename(file):
            print(f"Skipping file containing 'PSM3': {file}")
            continue
        if "SUJ" in os.path.basename(file):
            print(f"Skipping file containing 'SUJ': {file}")
            continue
        if "clutch" in os.path.basename(file):
            print(f"Skipping file containing 'SUJ': {file}")
            continue
        # if "/Volumes/drakes_ssd_500gb/skill_assessment/data/S01/parquet/T04" not in file:
        #     continue
        
        df = pd.read_parquet(file)
        
        # Preserve original timestamps as columns
        df["orig_header_stamp_sec"] = df["header_stamp_sec"]
        df["orig_header_stamp_nsec"] = df["header_stamp_nsec"]
        df["orig_datetime"] = pd.to_datetime(df["orig_header_stamp_sec"] + df["orig_header_stamp_nsec"] / 1e9, unit="s")
        

        # Compute timestamp for alignment
        df["timestamp"] = df["header_stamp_sec"] + df["header_stamp_nsec"] / 1e9
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

        
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.infer_objects(copy=False)

        # Prepare baseline DataFrame for merge_asof
        baseline_df_tmp = pd.DataFrame(index=baseline_index).reset_index().rename(columns={'index': 'datetime'})
        baseline_df_tmp["timestamp"] = baseline_df_tmp["datetime"].astype('int64') / 1e9

        # Align using merge_asof (nearest within tolerance)
        aligned_df = pd.merge_asof(
            baseline_df_tmp,
            df.reset_index(),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta(target_frequency)
        )
        aligned_df = aligned_df.set_index("datetime")

        # Save true stamps for original samples
        orig_sec  = aligned_df['header_stamp_sec'].copy() if 'header_stamp_sec' in aligned_df.columns else None
        orig_nsec = aligned_df['header_stamp_nsec'].copy() if 'header_stamp_nsec' in aligned_df.columns else None

        # Mask for exact original timestamps
        mask_exact = aligned_df.index.isin(df.index)

        # Data columns to process (all numeric sensor fields; exclude only timestamps and seq)
        data_cols = aligned_df.select_dtypes(include=['number']).columns.difference(
            ['header_stamp_sec', 'header_stamp_nsec', 'header_seq']
        )

        # For exact matches, copy original data values
        for col in data_cols:
            if col in df.columns:
                aligned_df.loc[mask_exact, col] = df.loc[aligned_df.index.intersection(df.index), col].values

        # For non-original timestamps, backward-fill first, then forward-fill (latch)
        aligned_df[data_cols] = aligned_df[data_cols].bfill().ffill()

        # Compute new timestamps from the upsampled index
        timestamps_ns = aligned_df.index.astype('int64')
        computed_sec  = (timestamps_ns // 1_000_000_000).astype(float)
        computed_nsec = (timestamps_ns %  1_000_000_000).astype(float)

        # Preserve original stamps on exact-sample rows, use computed for midpoints
        if orig_sec is not None:
            mask_exact = aligned_df.index.isin(df.index)
            aligned_df['header_stamp_sec']  = np.where(mask_exact, orig_sec,  computed_sec)
            aligned_df['header_stamp_nsec'] = np.where(mask_exact, orig_nsec, computed_nsec)
        else:
            aligned_df['header_stamp_sec']  = computed_sec
            aligned_df['header_stamp_nsec'] = computed_nsec

        # Reset header_seq as monotonic counter
        aligned_df['header_seq'] = np.arange(len(aligned_df))

        # Add BagFileName column
        aligned_df["BagFileName"] = os.path.basename(file)

        # Calculate alignment error (residual) for original timestamps to nearest baseline timestamp
        orig_times = pd.to_datetime(df["orig_datetime"], errors='coerce').dropna()
        baseline_times = pd.DatetimeIndex(baseline_index)

        def nearest_residual(ts):
            if pd.isnull(ts):
                return None
            idx = baseline_times.get_indexer([ts], method="nearest")[0]
            nearest = baseline_times[idx]
            return abs((ts - nearest).total_seconds())

        residuals = orig_times.apply(nearest_residual).dropna()
        mean_residual = residuals.mean()
        alignment_errors[file] = mean_residual

        print(f"Mean alignment error for {os.path.basename(file)}: {mean_residual:.6f} seconds")


        # Drop unwanted merge artifacts and internal columns, but always preserve BagFileName
        aligned_df = aligned_df.drop(
            columns=[c for c in aligned_df.columns if (c.startswith('timestamp_') or c.startswith('orig_') or c == 'datetime') and c != 'BagFileName'],
            errors='ignore'
        )

        # Preserve original header order
        expected_first = ['header_seq', 'header_stamp_sec', 'header_stamp_nsec', 'header_frame_id']
        expected_last = ['BagFileName']
        first_cols = [c for c in expected_first if c in aligned_df.columns]
        last_cols = [c for c in expected_last if c in aligned_df.columns]
        middle_cols = [c for c in aligned_df.columns if c not in first_cols + last_cols]
        aligned_df = aligned_df[first_cols + middle_cols + last_cols]

        # Ensure header_frame_id is cleanly typed
        if 'header_frame_id' in aligned_df.columns:
            aligned_df['header_frame_id'] = pd.to_numeric(aligned_df['header_frame_id'], errors='coerce').fillna(0).astype('int64')

        # Save aligned DataFrame
        output_file = os.path.join(output_dir, os.path.basename(file))
        aligned_df.to_parquet(output_file)
        print(f"Saved aligned and downsampled file: {output_file}")

    return alignment_errors

# --- Main script ---

path_to_write_data = "preprocessed_data"

# Creating directory called data preprocessed in the top level of the repo
path_to_write_data_ = os.path.join(path_to_write_data)

# print("is this the error")
if os.path.exists(path_to_write_data_):
    # os.rmdir(path_to_write_data_)
    # remove dir and all of its contents
    shutil.rmtree(path_to_write_data_)


os.mkdir(path_to_write_data_)


path_to_data = "/Volumes/drakes_ssd_500gb/skill_assessment/data/"

subject_dirs= find_subject_files(path_to_data)
all_entries = os.listdir(path_to_data)

# print(subject_dirs)

# exit()

for subject_dir in subject_dirs:

    # Progress Bar for Preprocess
    pbar = tqdm(desc="Subject " + subject_dir)

    # loading all trial dir with parquet files
    path_to_parquet_files = os.path.join(path_to_data, subject_dir, "parquet")
    # print(path_to_parquet_files)
    trial_dirs = [d for d in os.listdir(path_to_parquet_files) if not d.startswith('.')]


    # print(trial_dirs)
    pbar.total = len(trial_dirs)

    # iterating through each trial per subject
    for trial_count, trial_dir in enumerate(trial_dirs):

        new_df = pd.DataFrame()

        parquet_files = [
            os.path.join(path_to_parquet_files, trial_dir, f)
            for f in os.listdir(os.path.join(path_to_parquet_files, trial_dir))
            if os.path.isfile(os.path.join(path_to_parquet_files, trial_dir, f))
            and ".parquet" in f
            and "console" not in f
            and "select" not in f
            and f != ".DS_Store"
        ]

        # print(os.path.join(path_to_parquet_files, trial_dir))
        # for file in parquet_files:
        #     print(file)

        output_dir = os.path.join(path_to_write_data, subject_dir, trial_dir)

        print("##### ALIGNING SUBECT: ", subject_dir, " TRIAL: ", trial_dir)

        alignment_errors = align_and_downsample(
            parquet_files=parquet_files,
            baseline_file=parquet_files[0],
            target_frequency = "5.291005ms",
            output_dir=output_dir
        )
        
    pbar.update()

        # Print alignment errors
        # for file, error in alignment_errors.items():
        #     print(f"Mean alignment error for {os.path.basename(file)}: {error:.6f} seconds")


