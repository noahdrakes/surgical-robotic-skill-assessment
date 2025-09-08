import pandas as pd
import numpy as np
import os
from preprocess import find_subject_files
from trial_valid import is_trial_valid
from tqdm import tqdm
import shutil
import json

# --- Timestamp computation and cropping functions ---

def _compute_timestamps(df):
    """
    Compute combined integer nanosecond timestamps from header seconds and nanoseconds columns.

    Args:
        df (pd.DataFrame): DataFrame with 'header_stamp_sec' and 'header_stamp_nsec' columns.

    Returns:
        pd.DataFrame: DataFrame with new 'ts_ns' column representing timestamps in nanoseconds.
    """
    sec_int = df["header_stamp_sec"].astype("int64")
    nsec_int = df["header_stamp_nsec"].astype("int64")
    ts_ns = sec_int * 1_000_000_000 + nsec_int
    df["ts_ns"] = ts_ns
    return df

def _crop_dataframe(df, crop_start, crop_end):
    """
    Crop DataFrame rows based on index values within the specified time range.

    Args:
        df (pd.DataFrame): DataFrame indexed by nanosecond timestamps.
        crop_start (pd.Timestamp or None): Start time for cropping.
        crop_end (pd.Timestamp or None): End time for cropping.

    Returns:
        pd.DataFrame: Cropped DataFrame.
    """
    if crop_start and crop_end:
        start_ns = int(crop_start.value)
        end_ns = int(crop_end.value)
        df = df[(df.index >= start_ns) & (df.index <= end_ns)]
    return df

# --- Index creation and preparation for alignment ---

def _create_upsampled_index(start, end, freq):
    """
    Create a datetime index including baseline timestamps and midpoints for upsampling.

    Args:
        start (pd.Timestamp): Start timestamp.
        end (pd.Timestamp): End timestamp.
        freq (str): Frequency string (e.g., '20.833ms').

    Returns:
        pd.DatetimeIndex: Sorted union of baseline and midpoint timestamps.
    """
    baseline_index = pd.date_range(start=start, end=end, freq=freq)
    midpoints = baseline_index[:-1] + (baseline_index[1:] - baseline_index[:-1]) / 2
    upsampled_index = baseline_index.union(midpoints).sort_values()
    return upsampled_index

def _prepare_baseline_for_alignment(baseline_index, target_frequency):
    """
    Prepare a baseline DataFrame with integer nanosecond timestamps and float seconds for alignment.

    Args:
        baseline_index (pd.DatetimeIndex): Index of timestamps.
        target_frequency (str): Frequency string (unused but kept for interface consistency).

    Returns:
        pd.DataFrame: DataFrame with 'ts_ns' and 'timestamp' columns for alignment.
    """
    baseline_df_tmp = (
        pd.DataFrame(index=baseline_index)
          .reset_index()
          .rename(columns={"index": "ts_ns"})
    )
    baseline_df_tmp["timestamp"] = baseline_df_tmp["ts_ns"].astype("int64") / 1e9
    return baseline_df_tmp

# --- Alignment functions ---

def _align_to_baseline(baseline_df_tmp, df, target_frequency):
    """
    Align DataFrame to baseline timestamps using nearest merge within tolerance.

    Args:
        baseline_df_tmp (pd.DataFrame): Baseline DataFrame with 'ts_ns' column.
        df (pd.DataFrame): DataFrame indexed by nanosecond timestamps.
        target_frequency (str): Frequency string for tolerance in merge_asof.

    Returns:
        pd.DataFrame: Aligned DataFrame indexed by 'ts_ns'.
    """
    aligned_df = pd.merge_asof(
        baseline_df_tmp,
        df.reset_index(),
        on="ts_ns",
        direction="nearest",
        tolerance=pd.Timedelta(target_frequency),
        allow_exact_matches=True
    )
    aligned_df = aligned_df.set_index("ts_ns")
    aligned_df = aligned_df[~aligned_df.index.duplicated(keep="first")]
    return aligned_df

def _fill_missing_data(aligned_df, df):
    """
    Placeholder for missing data handling. Currently returns aligned DataFrame unchanged.

    Args:
        aligned_df (pd.DataFrame): Aligned DataFrame possibly containing NaNs.
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: Unchanged aligned DataFrame.
    """
    return aligned_df

def _calculate_alignment_error(df, baseline_index):
    """
    Calculate mean absolute alignment error in seconds between original and baseline timestamps.

    Args:
        df (pd.DataFrame): Original DataFrame indexed by nanosecond timestamps.
        baseline_index (pd.DatetimeIndex or iterable): Baseline timestamps in seconds.

    Returns:
        float: Mean residual error in seconds, or NaN if no valid timestamps.
    """
    residuals = []
    for ts in df.index:
        try:
            ts_float = ts / 1e9
            nearest = min(baseline_index, key=lambda x: abs(x - ts_float))
            residuals.append(abs(ts_float - nearest))
        except Exception:
            continue
    if len(residuals) == 0:
        return np.nan
    else:
        return np.mean(residuals)

# --- Finalization and saving ---

def _finalize_and_save_dataframe(aligned_df, file, output_dir):
    """
    Finalize DataFrame by cleaning columns, enforcing types, and saving as Parquet.

    Args:
        aligned_df (pd.DataFrame): Aligned DataFrame to save.
        file (str): Original file path for naming output.
        output_dir (str): Directory to save the output file.

    Returns:
        None
    """
    # Ensure 'header_seq' exists
    if 'header_seq' not in aligned_df.columns:
        aligned_df['header_seq'] = np.arange(len(aligned_df))

    # Drop unwanted columns except 'BagFileName'
    aligned_df = aligned_df.drop(
        columns=[
            c for c in aligned_df.columns
            if (
                (c.startswith('timestamp_'))
                or (c.startswith('orig_') and not c.startswith('orig_header_stamp'))
                or c == 'datetime'
            )
            and c != 'BagFileName'
        ],
        errors='ignore'
    )

    # Reorder columns: header_seq, ts_ns, header_frame_id first; BagFileName last
    expected_first = ['header_seq', 'ts_ns', 'header_frame_id']
    expected_last = ['BagFileName']
    first_cols = [c for c in expected_first if c in aligned_df.columns]
    last_cols = [c for c in expected_last if c in aligned_df.columns]
    middle_cols = [c for c in aligned_df.columns if c not in first_cols + last_cols]
    aligned_df = aligned_df[first_cols + middle_cols + last_cols]

    # Clean 'header_frame_id' type
    if 'header_frame_id' in aligned_df.columns:
        aligned_df['header_frame_id'] = pd.to_numeric(aligned_df['header_frame_id'], errors='coerce').fillna(0).astype('int64')

    # Save to Parquet
    output_file = os.path.join(output_dir, os.path.basename(file))
    aligned_df.to_parquet(output_file)
    print(f"Saved aligned and downsampled file: {output_file}")

# --- Main alignment function ---

def align_and_downsample(
    parquet_files,
    baseline_file,
    target_frequency=None,
    output_dir="aligned_data",
    crop_start=None,
    crop_end=None
):
    """
    Align and optionally downsample Parquet files to a baseline file's timestamps.

    Args:
        parquet_files (list of str): List of Parquet file paths to align.
        baseline_file (str): Path to baseline Parquet file for reference timestamps.
        target_frequency (str or None): Frequency string for downsampling (e.g., '20.833ms'). If None, no resampling.
        output_dir (str): Directory to save aligned files.
        crop_start (pd.Timestamp or None): Start time to crop baseline and data.
        crop_end (pd.Timestamp or None): End time to crop baseline and data.

    Returns:
        dict: Mapping of file paths to mean alignment errors in seconds.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare baseline DataFrame with nanosecond timestamps
    baseline_df = pd.read_parquet(baseline_file)
    sec_int = baseline_df["header_stamp_sec"].astype("int64")
    nsec_int = baseline_df["header_stamp_nsec"].astype("int64")
    baseline_df["ts_ns"] = sec_int * 1_000_000_000 + nsec_int
    baseline_df = baseline_df.set_index("ts_ns")

    # Crop baseline if crop times provided
    baseline_df = _crop_dataframe(baseline_df, crop_start, crop_end)

    # Prepare baseline index and DataFrame for alignment
    if target_frequency is not None:
        baseline_index = _create_upsampled_index(
            start=baseline_df.index.min(),
            end=baseline_df.index.max(),
            freq=target_frequency
        )
        baseline_df_tmp = _prepare_baseline_for_alignment(baseline_index, target_frequency)
    else:
        baseline_index = baseline_df.index
        baseline_df_tmp = None

    alignment_errors = {}

    for file in parquet_files:
        basename = os.path.basename(file)
        if any(skip in basename for skip in ["follow_mode", "jaw", "PSM3", "SUJ"]):
            print(f"Skipping file containing skip pattern: {file}")
            continue

        df = pd.read_parquet(file)

        # Preserve original timestamp columns for reference
        df["orig_header_stamp_sec"] = pd.to_numeric(df["header_stamp_sec"], errors="coerce")
        df["orig_header_stamp_nsec"] = pd.to_numeric(df["header_stamp_nsec"], errors="coerce")

        num_nats = df["orig_header_stamp_sec"].isna().sum() + df["orig_header_stamp_nsec"].isna().sum()
        if num_nats > 0:
            print(f"[WARN] {num_nats} NaN values in original header stamp for file: {file}")

        # If no target frequency, skip alignment and just preserve timestamps
        if target_frequency is None:
            sec_int = df["orig_header_stamp_sec"].astype("int64")
            nsec_int = df["orig_header_stamp_nsec"].astype("int64")
            df["ts_ns"] = sec_int * 1_000_000_000 + nsec_int
            df = df.drop(columns=["header_stamp_sec", "header_stamp_nsec"], errors="ignore")
            df = df[~df["ts_ns"].duplicated(keep="first")]
            aligned_df = df.copy()
            _finalize_and_save_dataframe(aligned_df, file, output_dir)
            continue

        # Compute timestamps and set index for alignment
        df = _compute_timestamps(df)
        df = df.set_index("ts_ns")
        df = df[~df.index.duplicated(keep="first")]

        aligned_df = _align_to_baseline(baseline_df_tmp, df, target_frequency)
        aligned_df = _fill_missing_data(aligned_df, df)

        # Calculate and report alignment error
        mean_residual = _calculate_alignment_error(df, baseline_index)
        if np.isnan(mean_residual):
            print(f"[WARN] Skipping residual error for {file} — not enough valid timestamps")
        else:
            print(f"Mean alignment error for {basename}: {mean_residual:.6f} seconds")
        alignment_errors[file] = mean_residual

        # Finalize and save aligned DataFrame
        _finalize_and_save_dataframe(aligned_df, file, output_dir)

    return alignment_errors

# --- Utility function for crop times ---

def get_crop_times(subject_id: str, trial_id: str, crop_df):
    """
    Retrieve crop start and end times for a given subject and trial from crop DataFrame.

    Args:
        subject_id (str): Subject identifier.
        trial_id (str): Trial identifier (e.g., 'T1').
        crop_df (pd.DataFrame): DataFrame indexed by (subject, trial) with 'start_time' and 'end_time'.

    Returns:
        tuple: (crop_start, crop_end) as pd.Timestamp or (None, None) if not found.
    """
    try:
        trial_int = int(trial_id.lstrip("T"))
        row = crop_df.loc[(subject_id, trial_int)]
        return pd.to_datetime(row["start_time"], unit="s"), pd.to_datetime(row["end_time"], unit="s")
    except Exception:
        print(f"[WARN] No crop times found for {subject_id} trial {trial_id}")
        return None, None
    

# --- Main script execution ---

# Load crop times from CSV and set multi-index for lookup
CROPPED_TIMES_PATH = os.path.join(os.path.dirname(__file__), "croppedTimes.csv")
crop_df = pd.read_csv(CROPPED_TIMES_PATH, header=None, names=["subject", "trial", "start_time", "end_time"])
crop_df.set_index(["subject", "trial"], inplace=True)

path_to_write_data = "preprocessed_data"
path_to_write_data_ = os.path.join(path_to_write_data)

# Remove existing output directory if present and create fresh
if os.path.exists(path_to_write_data_):
    shutil.rmtree(path_to_write_data_)
os.mkdir(path_to_write_data_)

path_to_data = "/Volumes/drakes_ssd_500gb/skill_assessment/data/"

# Find subject directories containing data
subject_dirs = find_subject_files(path_to_data)

# Iterate over subjects and their trials to align and save data
for subject_dir in subject_dirs:

    # Initialize progress bar for subject
    pbar = tqdm(desc="Subject " + subject_dir)

    path_to_parquet_files = os.path.join(path_to_data, subject_dir, "parquet")

    # List trial directories for subject, ignoring hidden files
    trial_dirs = [d for d in os.listdir(path_to_parquet_files) if not d.startswith('.')]

    pbar.total = len(trial_dirs)

    for trial_count, trial_dir in enumerate(trial_dirs):

        valid_trials_csv = "/Users/noahdrakes/Documents/research/skill_assessment/MISTIC_robotic_suturing_study/protocol/trial_inclusion_matrix.csv"

        if is_trial_valid(valid_trials_csv, subject_dir, trial_dir) != True:
            print("SKIPPING INVALID TRIAL")
            continue

        # Gather Parquet files for the trial, filtering unwanted files
        parquet_files = [
            os.path.join(path_to_parquet_files, trial_dir, f)
            for f in os.listdir(os.path.join(path_to_parquet_files, trial_dir))
            if os.path.isfile(os.path.join(path_to_parquet_files, trial_dir, f))
            and ".parquet" in f
            and "console" not in f
            and "select" not in f
            and f != ".DS_Store"
        ]

        output_dir = os.path.join(path_to_write_data, subject_dir, trial_dir)

        print("##### ALIGNING SUBJECT:", subject_dir, "TRIAL:", trial_dir)

        # Retrieve cropping times for current subject and trial
        crop_start, crop_end = get_crop_times(subject_dir, trial_dir, crop_df)

        # Align and downsample files with cropping and save results
        alignment_errors = align_and_downsample(
            parquet_files=parquet_files,
            baseline_file=parquet_files[0],
            target_frequency=None,
            output_dir=output_dir,
            crop_start=crop_start,
            crop_end=crop_end
        )
        
    pbar.update()
