import pandas as pd
import os 
from trial_valid import is_trial_valid
import shutil
from tqdm import tqdm

def find_subject_files(path_to_data):
        return [f for f in os.listdir(path_to_data) if not f.startswith('.')]

file = "/home/ndrakes1/surgical_skill_assessment/MISTIC_robotic_suturing_study/protocol/feedback_matrix.csv"

df = pd.read_csv(file)


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

path_to_data = "preprocessed_data"

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

        valid_trials_csv = "/home/ndrakes1/surgical_skill_assessment/MISTIC_robotic_suturing_study/protocol/trial_inclusion_matrix.csv"

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

        print(parquet_files)

        output_dir = os.path.join(path_to_write_data, subject_dir, trial_dir)

        print("##### ALIGNING SUBJECT:", subject_dir, "TRIAL:", trial_dir)



