import pandas as pd
import numpy as np
import os
from preprocess import find_subject_files
from tqdm import tqdm
import shutil
import json

def _compute_timestamps(df):
    """
    Compute combined timestamp columns and datetime index.
    Store timestamp_float as relative to the first timestamp (float64).
    """
    # Build integer nanosecond timestamp and compute relative float seconds
    sec_int = df["header_stamp_sec"].astype("int64")
    nsec_int = df["header_stamp_nsec"].astype("int64")
    ts_ns = sec_int * 1_000_000_000 + nsec_int
    df["ts_ns"] = ts_ns
    # Relative timestamp_float in seconds
    delta_ns = ts_ns - ts_ns.iloc[0]
    df["timestamp_float"] = delta_ns.astype("float64") / 1e9
    return df

def _crop_dataframe(df, crop_start, crop_end):
    """
    Crop dataframe to the specified time range.
    """
    if crop_start and crop_end:
        start_ns = int(crop_start.value)
        end_ns = int(crop_end.value)
        df = df[(df.index >= start_ns) & (df.index <= end_ns)]
    return df

def _create_upsampled_index(start, end, freq):
    """
    Create a datetime index with upsampled midpoints between timestamps.
    """
    baseline_index = pd.date_range(start=start, end=end, freq=freq)
    midpoints = baseline_index[:-1] + (baseline_index[1:] - baseline_index[:-1]) / 2
    upsampled_index = baseline_index.union(midpoints).sort_values()
    return upsampled_index

def _prepare_baseline_for_alignment(baseline_index, target_frequency):
    """
    Prepare a baseline DataFrame for merge_asof alignment.
    """
    baseline_df_tmp = (
        pd.DataFrame(index=baseline_index)
          .reset_index()
          .rename(columns={"index": "ts_ns"})
    )
    baseline_df_tmp["timestamp"] = baseline_df_tmp["ts_ns"].astype("int64") / 1e9
    return baseline_df_tmp

def _align_to_baseline(baseline_df_tmp, df, target_frequency):
    """
    Align df to the baseline using merge_asof with nearest matching within tolerance.
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
    # Remove duplicate timestamps to avoid repeated samples
    aligned_df = aligned_df[~aligned_df.index.duplicated(keep="first")]
    return aligned_df

def _reconstruct_timestamps(aligned_df, df):
    """
    Reconstruct timestamp_float using the integer nanosecond index.
    """
    if df.index.duplicated().any():
        print(f"[WARN] Original parquet has repeating timestamps: {df.get('BagFileName', 'Unknown')}")
    if aligned_df.empty:
        aligned_df["timestamp_float"] = []
        return aligned_df
    # Use ts_ns index (int64 nanoseconds)
    ts_ns = aligned_df.index.astype("int64")
    first_ns = ts_ns[0]
    delta_ns = ts_ns - first_ns
    aligned_df["timestamp_float"] = delta_ns.astype("float64") / 1e9
    return aligned_df

def _fill_missing_data(aligned_df, df):
    """
    Do not fill missing data. Return the aligned DataFrame unchanged.
    """
    return aligned_df

def _clean_string_and_array_columns(aligned_df):
    """
    Clean string columns by removing newline characters and serialize array-like columns to JSON strings.
    """
    # Remove newline characters from string columns
    string_cols = aligned_df.select_dtypes(include=['object']).columns
    aligned_df[string_cols] = aligned_df[string_cols].map(
        lambda v: v.replace('\n', ' ') if isinstance(v, str) else v
    )

    # Serialize array-like columns to JSON strings
    array_cols = [
        col for col in aligned_df.columns
        if aligned_df[col].apply(lambda v: hasattr(v, 'tolist')).any()
    ]
    for col in array_cols:
        aligned_df[col] = aligned_df[col].apply(
            lambda v: json.dumps(v.tolist(), separators=(',', ':')) if hasattr(v, 'tolist') else v
        )
    return aligned_df

def _calculate_alignment_error(df, baseline_index):
    """
    Calculate mean alignment error (residual) between original timestamps and nearest baseline timestamps.
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

def _finalize_and_save_dataframe(aligned_df, file, output_dir):
    """
    Finalize dataframe by resetting sequence, organizing columns, cleaning types, and saving parquet file.
    """
    # Preserve original header_seq if it exists
    if 'header_seq' not in aligned_df.columns:
        aligned_df['header_seq'] = np.arange(len(aligned_df))


    # Drop unwanted merge artifacts and internal columns except BagFileName
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
    # Always drop header_stamp_sec and header_stamp_nsec if they exist
    # aligned_df = aligned_df.drop(columns=["header_stamp_sec", "header_stamp_nsec"], errors="ignore")

    # Preserve original header order
    expected_first = ['header_seq', 'ts_ns', 'header_frame_id']
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

def align_and_downsample(
    parquet_files,
    baseline_file,
    target_frequency=None,
    output_dir="aligned_data",
    crop_start=None,
    crop_end=None
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

    # Load baseline file and compute timestamps
    baseline_df = pd.read_parquet(baseline_file)
    # Build integer nanosecond baseline index
    sec_int = baseline_df["header_stamp_sec"].astype("int64")
    nsec_int = baseline_df["header_stamp_nsec"].astype("int64")
    baseline_df["ts_ns"] = sec_int * 1_000_000_000 + nsec_int
    baseline_df = baseline_df.set_index("ts_ns")

    # Crop baseline dataframe if requested
    baseline_df = _crop_dataframe(baseline_df, crop_start, crop_end)

    # Prepare the baseline index and baseline_df_tmp according to target_frequency
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

        # Preserve original timestamps as columns
        df["orig_header_stamp_sec"] = pd.to_numeric(df["header_stamp_sec"], errors="coerce")
        df["orig_header_stamp_nsec"] = pd.to_numeric(df["header_stamp_nsec"], errors="coerce")

        num_nats = df["orig_header_stamp_sec"].isna().sum() + df["orig_header_stamp_nsec"].isna().sum()
        if num_nats > 0:
            print(f"[WARN] {num_nats} NaN values in original header stamp for file: {file}")

        if target_frequency is None:
            # Preserve raw nanosecond timestamp as ts_ns
            sec_int = df["orig_header_stamp_sec"].astype("int64")
            nsec_int = df["orig_header_stamp_nsec"].astype("int64")
            df["ts_ns"] = sec_int * 1_000_000_000 + nsec_int
            # Drop original stamp and float timestamp columns
            df = df.drop(columns=["header_stamp_sec", "header_stamp_nsec", "timestamp_float"], errors="ignore")
            # Remove duplicate nanosecond timestamps
            df = df[~df["ts_ns"].duplicated(keep="first")]
            aligned_df = df.copy()
            _finalize_and_save_dataframe(aligned_df, file, output_dir)
            # Skip further processing for this file
            continue

        # Compute timestamps for alignment
        df = _compute_timestamps(df)
        df = df.set_index("ts_ns")  # use integer nanosecond index consistently
        # Remove any duplicate nanosecond indices
        df = df[~df.index.duplicated(keep="first")]

        # Crop dataframe if requested
        df = _crop_dataframe(df, crop_start, crop_end)

        aligned_df = _align_to_baseline(baseline_df_tmp, df, target_frequency)
        aligned_df = _fill_missing_data(aligned_df, df)

        # Reconstruct timestamps preserving original where possible
        aligned_df = _reconstruct_timestamps(aligned_df, df)

        # Clean string and array-like columns
        aligned_df = _clean_string_and_array_columns(aligned_df)

        # Calculate alignment error
        mean_residual = _calculate_alignment_error(df, baseline_index)
        if np.isnan(mean_residual):
            print(f"[WARN] Skipping residual error for {file} — not enough valid timestamps")
        else:
            print(f"Mean alignment error for {basename}: {mean_residual:.6f} seconds")
        alignment_errors[file] = mean_residual

        # Finalize dataframe and save
        _finalize_and_save_dataframe(aligned_df, file, output_dir)

    return alignment_errors

def get_crop_times(subject_id: str, trial_id: str, crop_df):
    try:
        trial_int = int(trial_id.lstrip("T"))
        row = crop_df.loc[(subject_id, trial_int)]
        return pd.to_datetime(row["start_time"], unit="s"), pd.to_datetime(row["end_time"], unit="s")
    except Exception:
        print(f"[WARN] No crop times found for {subject_id} trial {trial_id}")
        return None, None
    

# --- Main script ---

CROPPED_TIMES_PATH = os.path.join(os.path.dirname(__file__), "croppedTimes.csv")
crop_df = pd.read_csv(CROPPED_TIMES_PATH, header=None, names=["subject", "trial", "start_time", "end_time"])
crop_df.set_index(["subject", "trial"], inplace=True)


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

        # if "2" not in trial_dir:
        #     continue

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
        crop_start, crop_end = get_crop_times(subject_dir, trial_dir, crop_df)

        alignment_errors = align_and_downsample(
            parquet_files=parquet_files,
            baseline_file=parquet_files[0],
            # target_frequency = "5.291005ms",
            target_frequency=None,
            output_dir=output_dir,
            crop_start=crop_start,
            crop_end=crop_end
        )
        
    pbar.update()

        # Print alignment errors
        # for file, error in alignment_errors.items():
        #     print(f"Mean alignment error for {os.path.basename(file)}: {error:.6f} seconds")
