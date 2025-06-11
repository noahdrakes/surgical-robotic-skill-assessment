import numpy as np
import math
import csv
import json
import pandas as pd

class Metrics:

    def __init__(self, subject, num_trials):
        self.name = subject  # Instance attribute
        self.num_trials = num_trials 

        # Dictionary to map metric names to methods
        self.metric_functions = {
            "completion_time": self.compute_completion_time,
            "average_speed_magnitude": self.compute_average_speed_magnitude,
            "average_acceleration_magnitude": self.compute_average_acceleration_magnitude
        }

    def compute_completion_time(self, trial_data, config):
        # print(config["timestamp_axis"])
        timestamps = trial_data[config["timestamp_axis"]]
        return float(timestamps.iloc[-1] - timestamps.iloc[0])

    def compute_average_speed_magnitude(self, trial_data, config):
        speed_magnitude = 0
        for axis in config["axes"]:
            speed_avg = trial_data[axis].mean()
            speed_magnitude += pow(speed_avg, 2)
        return float(math.sqrt(speed_magnitude))

    def compute_average_acceleration_magnitude(self, trial_data, config):
        accel_sqrd = 0
        timestamps = trial_data[config["timestamp_axis"]]
        dt = np.diff(timestamps)

        if not config["using_accel"]:
            for axis in config["velocity_axes"]:
                velocity = trial_data[axis]
                acceleration = np.diff(velocity) / dt
                accel_sqrd += pow(acceleration.mean(), 2)
        else:
            for axis in config["accel_axes"]:
                acceleration = trial_data[axis]
                accel_sqrd += pow(acceleration.mean(), 2)

        return float(math.sqrt(accel_sqrd))

    def compute_metric(self, metric_name, trial_data, config):
        if metric_name == "completion_time":
            return self.compute_completion_time(trial_data, config)
        elif metric_name == "average_speed_magnitude":
            return self.compute_average_speed_magnitude(trial_data, config)
        elif metric_name == "average_acceleration_magnitude":
            return self.compute_average_acceleration_magnitude(trial_data, config)
        else:
            raise ValueError(f"Metric '{metric_name}' is not supported.")


class MetricsManager:
    def __init__(self, config_path):
        self.subjects = {}  # Dictionary to store trials for each subject

        # Load configuration from JSON file
        with open(config_path, "r") as f:
            self.metric_config = json.load(f)

    def add_subject(self, subject_name, num_trials):
        if subject_name not in self.subjects:
            self.subjects[subject_name] = {"metrics": Metrics(subject_name, num_trials), "trial_paths": []}

    def add_trial_path(self, subject_name, trial_path):
        if subject_name not in self.subjects:
            raise ValueError(f"Subject {subject_name} not found. Please add the subject first.")
        self.subjects[subject_name]["trial_paths"].append(trial_path)

    def load_trial_data(self, trial_path, header_path):
        """
        Load preprocessed trial data based on the file format (CSV or Parquet).

        Args:
            trial_path (str): Path to the trial's data file.
            header_path (str): Path to the rostopics header file.

        Returns:
            pd.DataFrame: DataFrame containing the trial data with proper column names.
        """
        import os

        # Determine file extension
        file_extension = os.path.splitext(trial_path)[1].lower()

        # Load header
        with open(header_path, "r") as f:
            header = f.readline().strip().split(",")

        # Load trial data based on file type
        if file_extension == ".csv":
            trial_data = pd.read_csv(trial_path, header=None)
        elif file_extension == ".parquet":
            trial_data = pd.read_parquet(trial_path, engine="pyarrow")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Assign header to columns
        trial_data.columns = header

        return trial_data

    def compute_metric_for_subject(self, subject_name, metric_name, header_path):
        if subject_name not in self.subjects:
            raise ValueError(f"No trials found for subject: {subject_name}")

        metrics = self.subjects[subject_name]["metrics"]
        trial_paths = self.subjects[subject_name]["trial_paths"]

        results = []
        for trial_path in trial_paths:
            trial_data_frame = self.load_trial_data(trial_path, header_path)

            # Check if the metric has sub-configurations (e.g., PSM-specific entries)
            if isinstance(self.metric_config[metric_name], dict) and "ros_topic" in self.metric_config[metric_name]:
                # Single configuration (e.g., completion_time)
                config = self.metric_config[metric_name]
                metric_result = metrics.compute_metric(metric_name, trial_data_frame, config)
                results.append({"Trial": trial_path, "Result": metric_result})
            else:
                # Multiple sub-configurations (e.g., average_speed_magnitude for PSMs)
                trial_results = {}
                for psm, config in self.metric_config[metric_name].items():
                    trial_results[psm] = metrics.compute_metric(metric_name, trial_data_frame, config)
                results.append(trial_results)

        return results

    def export_metrics_to_csv(self, metric_names, output_csv_path, header_path):
        """
        Export computed metrics for all subjects and trials to a single CSV file.

        Args:
            metric_names (list): List of metric names to compute.
            output_csv_path (str): Path to save the CSV file.
            header_path (str): Path to the rostopics header file.
        """
        csv_data = []
        header = ["Subject", "Trial", "PSM", "Metric", "Value"]  # Updated header

        for subject_name, subject_data in self.subjects.items():
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):
                trial_data_frame = self.load_trial_data(trial_path, header_path)

                for metric_name in metric_names:
                    # Check if the metric has sub-configurations
                    if isinstance(self.metric_config[metric_name], dict) and "ros_topic" in self.metric_config[metric_name]:
                        # Single configuration (e.g., completion_time)
                        config = self.metric_config[metric_name]
                        metric_result = metrics.compute_metric(metric_name, trial_data_frame, config)
                        row = [subject_name, f"Trial {trial_idx + 1}", "N/A", metric_name, metric_result]
                        csv_data.append(row)
                    elif isinstance(self.metric_config[metric_name], dict):
                        # Multiple sub-configurations (e.g., average_speed_magnitude for PSMs)
                        for psm, config in self.metric_config[metric_name].items():
                            metric_result = metrics.compute_metric(metric_name, trial_data_frame, config)
                            row = [subject_name, f"Trial {trial_idx + 1}", psm, metric_name, metric_result]
                            csv_data.append(row)
                    else:
                        raise ValueError(f"Unsupported metric configuration for '{metric_name}'.")

        # Write to CSV
        df = pd.DataFrame(csv_data, columns=header)
        df.to_csv(output_csv_path, index=False)

        print(f"Metrics saved to {output_csv_path}")

    def export_all_metrics_to_csv(self, metric_names, output_csv_path, header_path):
        """
        Export all computed metrics for all subjects and trials to a single CSV file.

        Args:
            metric_names (list): List of metric names to compute.
            output_csv_path (str): Path to save the CSV file.
            header_path (str): Path to the rostopics header file.
        """
        csv_data = []
        header = ["Subject", "Trial", "PSM"] + metric_names  # CSV header with all metrics

        for subject_name, subject_data in self.subjects.items():
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):
                trial_data_frame = self.load_trial_data(trial_path, header_path)

                # Check if the metric has sub-configurations
                row = [subject_name, f"Trial {trial_idx + 1}"]  # Initialize row with subject and trial info

                for metric_name in metric_names:
                    if isinstance(self.metric_config[metric_name], dict) and "ros_topic" in self.metric_config[metric_name]:
                        # Single configuration (e.g., completion_time)
                        config = self.metric_config[metric_name]
                        metric_result = metrics.compute_metric(metric_name, trial_data_frame, config)
                        row.append("N/A")  # PSM is not applicable for single configuration metrics
                        row.append(metric_result)
                    else:
                        # Multiple sub-configurations (e.g., average_speed_magnitude for PSMs)
                        for psm, config in self.metric_config[metric_name].items():
                            metric_result = metrics.compute_metric(metric_name, trial_data_frame, config)
                            row.append(psm)
                            row.append(metric_result)

                csv_data.append(row)

        # Write to CSV
        df = pd.DataFrame(csv_data, columns=header)
        df.to_csv(output_csv_path, index=False)

        print(f"All metrics saved to {output_csv_path}")



