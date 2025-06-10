import numpy as np
import math
import csv

class Metrics:

    def __init__(self, subject, num_trials):
        self.name = subject  # Instance attribute
        self.num_trials = num_trials 

        self.completion_times = np.zeros(num_trials)
        self.average_speed_magnitude = np.zeros(num_trials)
        self.average_acceleration_magnitude = np.zeros(num_trials)

        # Dictionary to map metric names to methods
        self.metric_functions = {
            "completion_time": self.compute_completion_time,
            "average_speed_magnitude": self.compute_average_speed_magnitude,
            "average_acceleration_magnitude": self.compute_average_acceleration_magnitude
        }

    def compute_completion_time(self, data_frame, timestamp_axis):
        df = np.array(data_frame)
        time = df[:, timestamp_axis]
        return float(time[-1] - time[0])

    def compute_average_speed_magnitude(self, data_frame, speed_axis_array):
        df = np.array(data_frame)
        speed_magnitude = 0

        for axis in speed_axis_array:
            speed_avg = df[:, axis].mean()
            speed_magnitude += pow(speed_avg, 2)

        return float(math.sqrt(speed_magnitude))

    def compute_average_acceleration_magnitude(self, data_frame, velocity_axis_array, accel_axis_array, timestamp_axis, using_accel=False):
        df = np.array(data_frame)
        accel_sqrd = 0

        timestamps = df[:, timestamp_axis]
        dt = np.diff(timestamps)

  
        accelerations = []

        if not using_accel:
            for axis in velocity_axis_array:
                accelerations = []
                accelerations = np.diff(df[:, axis], axis=0) / dt
                accel_sqrd += pow(accelerations.mean(), 2)
        else:
            for axis in accel_axis_array:
                accelerations = df[:, axis]
                accel_sqrd += pow(accelerations.mean(), 2)

        return float(math.sqrt(accel_sqrd))

    def compute_metric(self, metric_name, *args, **kwargs):
        if metric_name not in self.metric_functions:
            raise ValueError(f"Metric '{metric_name}' is not supported.")
        return self.metric_functions[metric_name](*args, **kwargs)


class MetricsManager:
    def __init__(self):
        self.subjects = {}  # Dictionary to store trials for each subject

        # Configuration for metric-specific arguments
        self.metric_config = {
            "completion_time": {"timestamp_axis": 0},
            "average_speed_magnitude": {"speed_axis_array": [1, 2]},
            "average_acceleration_magnitude": {
                "velocity_axis_array": [1, 2],
                "accel_axis_array": [3, 4],
                "timestamp_axis": 0,
                "using_accel": False
            }
        }

    def add_subject(self, subject_name, num_trials):
        if subject_name not in self.subjects:
            self.subjects[subject_name] = {"metrics": Metrics(subject_name, num_trials), "trial_paths": []}

    def add_trial_path(self, subject_name, trial_path):
        if subject_name not in self.subjects:
            raise ValueError(f"Subject {subject_name} not found. Please add the subject first.")
        self.subjects[subject_name]["trial_paths"].append(trial_path)

    def load_trial_data(self, trial_path):
        import pandas as pd
        import os

        file_extension = os.path.splitext(trial_path)[1].lower()

        if file_extension == ".csv":
            return pd.read_csv(trial_path).values
        elif file_extension == ".parquet":
            return pd.read_parquet(trial_path).values
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def compute_metric_for_subject(self, subject_name, metric_name, *args, **kwargs):
        if subject_name not in self.subjects:
            raise ValueError(f"No trials found for subject: {subject_name}")

        metrics = self.subjects[subject_name]["metrics"]
        trial_paths = self.subjects[subject_name]["trial_paths"]

        results = []
        for trial_path in trial_paths:
            trial_data_frame = self.load_trial_data(trial_path)
            result = metrics.compute_metric(metric_name, trial_data_frame, *args, **kwargs)
            results.append(result)
        return results

    def compute_metric_for_all_subjects(self, metric_name, *args, **kwargs):
        all_results = {}
        for subject_name in self.subjects:
            all_results[subject_name] = self.compute_metric_for_subject(subject_name, metric_name, *args, **kwargs)
        return all_results
    
    def export_metrics_to_csv(self, metric_names, output_csv_path):
        """
        Export all computed metrics for all subjects and trials to a CSV file.

        Args:
            metric_names (list): List of metric names to compute.
            output_csv_path (str): Path to save the CSV file.
        """
        csv_data = []
        header = ["Subject", "Trial"] + metric_names  # CSV header

        for subject_name, subject_data in self.subjects.items():
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):
                trial_data_frame = self.load_trial_data(trial_path)
                row = [subject_name, f"Trial {trial_idx + 1}"]  # Add subject and trial identifiers

                for metric_name in metric_names:
                    # Retrieve metric-specific arguments from the configuration
                    metric_args = self.metric_config.get(metric_name, {})
                    metric_result = metrics.compute_metric(metric_name, trial_data_frame, **metric_args)
                    row.append(metric_result)  # Append the metric result for this trial

                csv_data.append(row)

        # Write to CSV
        with open(output_csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)  # Write the header
            writer.writerows(csv_data)  # Write the rows

        print(f"Metrics saved to {output_csv_path}")



