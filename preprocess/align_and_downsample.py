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
    """
    df["timestamp"] = df["header_stamp_sec"] + df["header_stamp_nsec"] / 1e9
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

def _crop_dataframe(df, crop_start, crop_end):
    """
    Crop dataframe to the specified datetime range.
    """
    if crop_start and crop_end:
        df = df[(df.index >= crop_start) & (df.index <= crop_end)]
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
    baseline_df_tmp = pd.DataFrame(index=baseline_index).reset_index().rename(columns={'index': 'datetime'})
    baseline_df_tmp["timestamp"] = baseline_df_tmp["datetime"].astype('int64') / 1e9
    return baseline_df_tmp

def _align_to_baseline(baseline_df_tmp, df, target_frequency):
    """
    Align df to the baseline using merge_asof with nearest matching within tolerance.
    """
    aligned_df = pd.merge_asof(
        baseline_df_tmp,
        df.reset_index(),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(target_frequency),
        allow_exact_matches=True
    )
    aligned_df = aligned_df.set_index("datetime")
    return aligned_df

def _reconstruct_timestamps(aligned_df, df):
    """
    Reconstruct header_stamp_sec and header_stamp_nsec columns preserving original timestamps when exact match.
    """
    orig_sec = aligned_df['header_stamp_sec'].copy() if 'header_stamp_sec' in aligned_df.columns else None
    orig_nsec = aligned_df['header_stamp_nsec'].copy() if 'header_stamp_nsec' in aligned_df.columns else None

    mask_exact = aligned_df.index.isin(df.index)

    timestamps_ns = aligned_df.index.astype('int64')
    computed_sec = (timestamps_ns // 1_000_000_000).astype(float)
    computed_nsec = (timestamps_ns % 1_000_000_000).astype(float)

    if orig_sec is not None:
        aligned_df['header_stamp_sec'] = np.where(mask_exact, orig_sec, computed_sec)
        aligned_df['header_stamp_nsec'] = np.where(mask_exact, orig_nsec, computed_nsec)
    else:
        aligned_df['header_stamp_sec'] = computed_sec
        aligned_df['header_stamp_nsec'] = computed_nsec

    return aligned_df

def _fill_missing_data(aligned_df, df):
    """
    Fill missing numeric and string data in aligned_df by latching values.
    """
    # Identify numeric data columns excluding timestamp and seq columns
    data_cols = aligned_df.select_dtypes(include=['number']).columns.difference(
        ['header_stamp_sec', 'header_stamp_nsec', 'header_seq']
    )

    # For exact matches, copy original data values
    mask_exact = aligned_df.index.isin(df.index)
    for col in data_cols:
        if col in df.columns:
            aligned_df.loc[mask_exact, col] = df.loc[aligned_df.index.intersection(df.index), col].values

    # Backward-fill then forward-fill numeric columns to latch data
    aligned_df[data_cols] = aligned_df[data_cols].bfill().ffill()

    # Latch string columns similarly
    string_cols = aligned_df.select_dtypes(include=['object']).columns
    aligned_df[string_cols] = aligned_df[string_cols].bfill().ffill()

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
    orig_times = pd.to_datetime(df["orig_datetime"], errors='coerce').dropna()
    baseline_times = pd.DatetimeIndex(baseline_index)

    def nearest_residual(ts):
        if pd.isnull(ts):
            return None
        idx = baseline_times.get_indexer([ts], method="nearest")[0]
        nearest = baseline_times[idx]
        return abs((ts - nearest).total_seconds())

    residuals = orig_times.apply(nearest_residual).dropna()
    if residuals.empty:
        return np.nan
    else:
        return residuals.mean()

def _finalize_and_save_dataframe(aligned_df, file, output_dir):
    """
    Finalize dataframe by resetting sequence, organizing columns, cleaning types, and saving parquet file.
    """
    # Reset header_seq as monotonic counter
    aligned_df['header_seq'] = np.arange(len(aligned_df))

    # Add BagFileName column
    aligned_df["BagFileName"] = os.path.basename(file)

    # Drop unwanted merge artifacts and internal columns except BagFileName
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

def align_and_downsample(
    parquet_files,
    baseline_file,
    target_frequency="2.6455ms",
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
    baseline_df["timestamp"] = baseline_df["header_stamp_sec"] + baseline_df["header_stamp_nsec"] / 1e9
    baseline_df["datetime"] = pd.to_datetime(baseline_df["timestamp"], unit="s")
    baseline_df = baseline_df.set_index("datetime").sort_index()

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
        baseline_df_tmp = pd.DataFrame({'datetime': baseline_index})
        baseline_df_tmp["timestamp"] = baseline_df_tmp["datetime"].astype('int64') / 1e9

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
        df["orig_datetime"] = pd.to_datetime(df["orig_header_stamp_sec"] + df["orig_header_stamp_nsec"] / 1e9, unit="s")

        num_nats = df["orig_datetime"].isna().sum()
        if num_nats > 0:
            print(f"[WARN] {num_nats} NaT values in orig_datetime for file: {file}")

        # Compute timestamps for alignment
        df = _compute_timestamps(df)
        df = df.set_index("datetime").sort_index()

        # Crop dataframe if requested
        df = _crop_dataframe(df, crop_start, crop_end)

        # Remove duplicated indices
        df = df[~df.index.duplicated(keep="first")]
        df = df.infer_objects(copy=False)

        # Align using merge_asof or use df directly if target_frequency is None
        if target_frequency is None:
            aligned_df = df.copy()
        else:
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
