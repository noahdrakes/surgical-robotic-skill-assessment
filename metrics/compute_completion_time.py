import os
import pandas as pd
import math
import argparse
import shutil
from tqdm import tqdm
from pprint import pprint
from Metrics import Metrics, MetricsManager

def find_subject_files(path_to_data):
    return [f for f in os.listdir(path_to_data) if not f.startswith('.')]


def add_all_subjects(manager, path_to_data):
    
    # Grabbing all of the subject directories
    subject_dirs = find_subject_files(path_to_data)

    print("Adding all Subjects...")

    # Iterating through each subject
    for subject_dir in subject_dirs:

        # Progress Bar for Preprocess
        pbar = tqdm(desc="Subject " + subject_dir)

        subject_path = os.path.join(path_to_data, subject_dir)

        # Loading all trial directories with parquet files
        subject_trials = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if os.path.isfile(os.path.join(subject_path, f))]

        num_trials = len(subject_trials)
        pbar.total = num_trials

        manager.add_subject(subject_dir, num_trials)

        for subject_trial in subject_trials:

            if os.path.splitext(subject_trial)[1] == ".txt":
                continue

            manager.add_trial_path(subject_dir, subject_trial)

            pbar.update()

## printing completion time
manager = MetricsManager(config_path="metric_config.json")

print("adding all subjects")
path_to_data = "/Users/noahdrakes/Documents/research/skill_assessment/surgical-robotic-skill-assessment/data_preprocessed"
add_all_subjects(manager, path_to_data)

print("subjects added")

header_path = "/Users/noahdrakes/Documents/research/skill_assessment/surgical-robotic-skill-assessment/data_preprocessed/S01/rostopics_header.txt"
output_csv_path = "results.csv"

print("exporting to csv")
metric_names = ["completion_time","average_speed_magnitude"]
manager.export_metrics_to_csv(metric_names, output_csv_path, header_path)
print("finished")

# print("Completion Time Results: ", completion_time_results)

