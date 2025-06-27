import numpy as np
import math
import csv
import json
import pandas as pd

completion_time_count = 0

class Metrics:
    def __init__(self, subject, num_trials):
        self.name = subject
        self.num_trials = num_trials

    def compute_completion_time(self, trial_data, config):
        timestamps = trial_data[config["timestamp_axis"]]

        if config["use_force_sensor"] == False:
            return float(timestamps.iloc[-1] - timestamps.iloc[0])
        
        ## CASE of when force sensor 
        if config["use_force_sensor"] == True:

            force_z_field = config["ATImini40"]["fields"][2]  ## 2 is hardcoded for the z axis
            force_z_data = trial_data[force_z_field] 

            force_z_magnitude = 0
            len_force_z_data = len(force_z_data)

            print(len_force_z_data)
            count = 0

            ### FIRST FORCE IMPULSE ## 
            while (count >= len_force_z_data):

                force_z_magnitude = force_z_data[count]

                if force_z_magnitude < -2 :
                    # print("mag: ", force_z_magnitude)
                    break

                count+=1
    
            index_of_first_button_press = count

            ## first button press timestamp
            first_button_press_timestamp = timestamps[index_of_first_button_press]

            # checking the last button press within 20 seconds of the first button press
            end_range = first_button_press_timestamp + 20

            last_button_press_idx = 0

            # print("index of first button press: ", index_of_first_button_press)
            # print("first press timestamp: ", timestamps[index_of_first_button_press])

            for i in range(index_of_first_button_press, len_force_z_data):

                force_z_magnitude = force_z_data[i]

                if force_z_magnitude < -2:
                    last_button_press_idx = i

                if timestamps[i] > end_range:
                    break
            
            # print("timestamp of last button press: ", timestamps[last_button_press_idx])

            # if (completion_time_count > 2):
            #     while(1):
            #         i =0
            

            count = len_force_z_data - 1
            

            ### LAST FORCE IMPULSE ###
            while (count >= len_force_z_data - 2000):

                force_z_magnitude = force_z_data[count]
                low_magnitude_idx = count

                if force_z_magnitude >= -.1 and force_z_magnitude <= .1 :
                    # print("mag: ", force_z_magnitude)
                    low_magnitude_idx = count
                    break

                count-=1
            
            print(low_magnitude_idx)

            return float(timestamps.iloc[-1] - timestamps.iloc[last_button_press_idx])
        
    def compute_average_speed_magnitude(self, trial_data, rostopic_config):
        speed_magnitude = 0
        for field in rostopic_config["fields"]:
            speed_avg = trial_data[field].mean()
            speed_magnitude += pow(speed_avg, 2)
        return float(math.sqrt(speed_magnitude))

    def compute_average_acceleration_magnitude(self, trial_data, rostopic_config):
        accel_sqrd = 0
        timestamps = trial_data[rostopic_config["timestamp_axis"]]
        dt = np.diff(timestamps)

        if not rostopic_config["using_accel"]:
            for field in rostopic_config["velocity_fields"]:
                velocity = trial_data[field]
                acceleration = np.diff(velocity) / dt
                accel_sqrd += pow(acceleration.mean(), 2)
        else:
            for field in rostopic_config["acceleration_fields"]:
                acceleration = trial_data[field]
                accel_sqrd += pow(acceleration.mean(), 2)

        return float(math.sqrt(accel_sqrd))

    def compute_metric(self, metric_name, trial_data, config):
        if metric_name == "completion_time":
            return self.compute_completion_time(trial_data, config)
        elif metric_name == "average_speed_magnitude":
            results = {}
            for rostopic, rostopic_config in config.items():
                results[rostopic] = self.compute_average_speed_magnitude(trial_data, rostopic_config)
            return results
        elif metric_name == "average_acceleration_magnitude":
            results = {}
            for rostopic, rostopic_config in config.items():
                results[rostopic] = self.compute_average_acceleration_magnitude(trial_data, rostopic_config)
            return results
        else:
            raise ValueError(f"Metric '{metric_name}' is not supported.")


class MetricsManager:
    def __init__(self, config_path):
        self.subjects = {}
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
        with open(header_path, "r") as f:
            header = f.readline().strip().split(",")
        trial_data = pd.read_csv(trial_path, header=None)
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
            config = self.metric_config[metric_name]
            metric_result = metrics.compute_metric(metric_name, trial_data_frame, config)
            results.append({"Trial": trial_path, "Result": metric_result})

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
        header = ["Subject", "Trial", "Metric", "Rostopic", "Value"]

        for subject_name, subject_data in self.subjects.items():
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):
                trial_data_frame = self.load_trial_data(trial_path, header_path)

                print(subject_name)
                print(trial_path)
                for metric_name in metric_names:
                    config = self.metric_config[metric_name]
                    metric_results = metrics.compute_metric(metric_name, trial_data_frame, config)

                    # Handle single-value metrics (e.g., completion_time)
                    if isinstance(metric_results, float):
                        row = [subject_name, f"Trial {trial_idx + 1}", metric_name, "N/A", metric_results]
                        csv_data.append(row)
                    # Handle multi-value metrics (e.g., average_speed_magnitude)
                    elif isinstance(metric_results, dict):
                        for rostopic, value in metric_results.items():
                            row = [subject_name, f"Trial {trial_idx + 1}", metric_name, rostopic, value]
                            csv_data.append(row)
                    else:
                        raise ValueError(f"Unsupported metric result type for '{metric_name}': {type(metric_results)}")

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
            print(subject_name)
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):
                trial_data_frame = self.load_trial_data(trial_path, header_path)

                # Check if the metric has sub-configurations
                row = [subject_name, f"Trial {trial_idx + 1}"]  # Initialize row with subject and trial info

                print(trial_path)
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



