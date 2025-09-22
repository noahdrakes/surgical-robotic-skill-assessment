import numpy as np
import math
import csv
import json
import pandas as pd
import os
from tqdm import tqdm

completion_time_count = 0

class Metrics:

    
    def __init__(self, subject, num_trials):
        self.name = subject
        self.num_trials = num_trials
        self.metric_fns = {
            "completion_time": self.compute_completion_time,
            "average_speed_magnitude": self.compute_average_speed_magnitude,
            "average_acceleration_magnitude": self.compute_average_acceleration_magnitude,
            "average_jerk_magnitude": self.compute_average_jerk_magnitude,
            "average_force_magnitude": self.compute_average_force_magnitude,
            "total_path_length": self.compute_total_path_length,
            "average_angular_speed_magnitude": self.compute_average_angular_speed_magnitude,
            "speed_correlation": self.compute_speed_correlation,
            "speed_cross": self.compute_speed_cross,
            "acceleration_cross": self.compute_acceleration_cross,
            "jerk_cross": self.compute_jerk_cross,
            "acceleration_dispertion": self.compute_accel_dispertion,
            "jerk_dispertion": self.compute_jerk_dispertion,
            "forcen_magnitude": self.compute_forcen_magnitude,
            # "acceleration_nT": self.compute_acceleration_nT
        }

    def compute_completion_time(self, dfs, config):
        # dfs is a dict of DataFrames keyed by filename
        # For completion_time, expect single file, get first file's df
        trial_data = list(dfs.values())[0]
        timestamps = trial_data[config["timestamp_col"]]
        
        end_time = float (timestamps.iloc[-1] / 1e9)
        start_time = float (timestamps.iloc[0] / 1e9)

        return end_time - start_time
        
    def compute_average_speed_magnitude(self, dfs, config):
        trial_data = list(dfs.values())[0]
        # compute speed magnitude at each timestep
        speed_sq = np.zeros(len(trial_data))
        for field in config["fields"]:
            speed_sq += trial_data[field] ** 2
        speed_magnitude = np.sqrt(speed_sq)
        return float(speed_magnitude.mean())

    def compute_average_acceleration_magnitude(self, dfs, config):
        # If using_accel is True and an accel file is present, use it
        if config["using_accel"] and len(dfs) > 1:
            # using dataframe that contains the word accel in it to use the accelerometer signal to 
            # do the calculation, as opposed to measured velocity
            accel_df = next((df for fname, df in dfs.items() if "accel" in fname.lower()), None)
            if accel_df is not None:
                # accel_sqrd = 0
                # for field in config["acceleration_fields"]:
                #     acceleration = accel_df[field]
                #     accel_sqrd += pow(acceleration.mean(), 2)
                accel_mag = self.__compute_magnitude(accel_df, config, "acceleration_fields")
                return np.mean(accel_mag)


                return float(math.sqrt(accel_sqrd))

        # Otherwise compute acceleration from velocity
        trial_data = list(dfs.values())[0]
        timestamps = trial_data[config["timestamp_col"]]
        dt = np.diff(timestamps)
        accel_sqrd = 0
        for field in config["velocity_fields"]:
            velocity = trial_data[field]
            acceleration = np.diff(velocity) / dt
            accel_sqrd += pow(acceleration.mean(), 2)

        return float(math.sqrt(accel_sqrd))

    def compute_average_jerk_magnitude(self, dfs, config):
        trial_data = list(dfs.values())[0]
        timestamps = trial_data[config["timestamp_col"]]

        dt = np.diff(timestamps)
        jerk_sqrd = 0

        for field in config["velocity_fields"]:
            velocity = trial_data[field]
            acceleration = np.diff(velocity) / dt
            dt_accel = dt[1:]
            jerk = np.diff(acceleration) / dt_accel
            jerk_sqrd += np.mean(jerk**2)
        
        return float(math.sqrt(jerk_sqrd))
    
    def compute_average_force_magnitude(self, dfs, config):
        trial_data = list(dfs.values())[0]
        # compute speed magnitude at each timestep
        force_sq = np.zeros(len(trial_data))
        for field in config["fields"]:
            force_sq += trial_data[field] ** 2
        force_magnitude = np.sqrt(force_sq)
        return float(force_magnitude.mean())
    
    def compute_total_path_length(self, dfs, config):
        trial_data = list(dfs.values())[0]

        field_diff_sqred = np.zeros(len(trial_data) - 1)
        for field in config["fields"]:
            field_diff_sqred += np.diff(trial_data[field]) ** 2
        
        return float(np.sqrt(field_diff_sqred).sum())
    
    def compute_average_angular_speed_magnitude(self, dfs, config):
        trial_data = list(dfs.values())[0]
        # compute speed magnitude at each timestep
        speed_sq = np.zeros(len(trial_data))
        for field in config["fields"]:
            speed_sq += trial_data[field] ** 2
        speed_magnitude = np.sqrt(speed_sq)
        return float(speed_magnitude.mean())

    def compute_speed_correlation(self, dfs, config):
        PSMa_speed, PSMb_speed = list(dfs.values())

        n = len(PSMa_speed) if (len(PSMa_speed) < len(PSMb_speed)) else len(PSMb_speed)

        PSMa_speed = PSMa_speed[:n]
        PSMb_speed = PSMb_speed[:n]

        PSMa_speed_magnitude = self.__compute_magnitude(PSMa_speed, config, "fields")
        PSMb_speed_magnitude = self.__compute_magnitude(PSMb_speed, config, "fields")

        r = np.corrcoef(PSMa_speed_magnitude, PSMb_speed_magnitude)[0][1]

        return r
    
    def __compute_magnitude(self, df, config, fields):
        sq = np.zeros(len(df))
        for field in config[fields]:
            sq += df[field] ** 2
        df_magnitude = np.sqrt(sq)
        return df_magnitude

    
    def __cross(self, x, y):
        if len(x) != len(y):
            print("[ERROR: __cross fn], arrays have mismatched lengths")
            return None
        
        xmin = x[0]
        ymin = y[0]

        xmin_upper = 0
        ymin_upper = 0

        xmin_upper += np.abs(xmin) * 1.05
        ymin_upper += np.abs(ymin) * 1.05

        # print("xmin: ", xmin, " xmin_upper: ", xmin_upper)
        # print("ymin: ", ymin, " ymin_upper: ", ymin_upper)

        x = np.asarray(x).astype(np.float64)
        y = np.asarray(y).astype(np.float64)

        counter = 0

        for i in range(len(x)):
            if x[i] >= xmin and x[i] < xmin_upper and y[i] >= ymin and y[i] < ymin_upper:
                counter += 1
        
        return float(counter) / float(len(x))


    def compute_speed_cross(self, dfs, config):
        PSMa_speed, PSMb_speed = list(dfs.values())

        n = len(PSMa_speed) if (len(PSMa_speed) < len(PSMb_speed)) else len(PSMb_speed)

        PSMa_speed = PSMa_speed[:n]
        PSMb_speed = PSMb_speed[:n]

        PSMa_speed_magnitude = self.__compute_magnitude(PSMa_speed, config,"fields")
        PSMb_speed_magnitude = self.__compute_magnitude(PSMb_speed, config, "fields")

        cross = self.__cross(PSMa_speed_magnitude, PSMb_speed_magnitude)
        # cross = self.__cross(linearx_a, linearx_b)
        return cross
    
    def compute_acceleration_cross(self, dfs, config):
        PSMa_accel, PSMb_accel = list(dfs.values())

        n = len(PSMa_accel) if (len(PSMa_accel) < len(PSMb_accel)) else len(PSMb_accel)

        PSMa_accel = PSMa_accel[:n]
        PSMb_accel = PSMb_accel[:n]

        PSMa_accel_magnitude = self.__compute_magnitude(PSMa_accel, config, "fields")
        PSMb_accel_magnitude = self.__compute_magnitude(PSMb_accel, config, "fields")

        cross = self.__cross(PSMa_accel_magnitude, PSMb_accel_magnitude)
        
        # return np.mean(PSMa_accel_magnitude)
        return cross


    def compute_jerk_cross(self, dfs, config):
        PSMa_vel, PSMb_vel = list(dfs.values())

        n = len(PSMa_vel) if (len(PSMa_vel) < len(PSMb_vel)) else len(PSMb_vel)

        PSMa_vel = PSMa_vel[:n]
        PSMb_vel = PSMb_vel[:n]

        ts_col = config["timestamp_col"]
        timestamps_a = PSMa_vel[ts_col].values
        timestamps_b = PSMb_vel[ts_col].values

        dt_a = np.diff(timestamps_a)
        dt_b = np.diff(timestamps_b)

        jerk_sqrd_a = np.zeros(n-2)
        jerk_sqrd_b = np.zeros(n-2)

        for field in config["fields"]:
            velocity = PSMa_vel[field]
            acceleration = np.diff(velocity) / dt_a
            dt_accel = dt_a[1:]
            jerk = np.diff(acceleration) / dt_accel
            jerk_sqrd_a += np.square(jerk)
        
        jerk_magnitude_a = np.sqrt(jerk_sqrd_a)

        for field in config["fields"]:
            velocity = PSMb_vel[field]
            acceleration = np.diff(velocity) / dt_b
            dt_accel = dt_b[1:]
            jerk = np.diff(acceleration) / dt_accel
            jerk_sqrd_b += np.square(jerk)
        
        jerk_magnitude_b = np.sqrt(jerk_sqrd_b)

        cross = self.__cross(jerk_magnitude_a, jerk_magnitude_b)
        return cross

    def compute_accel_dispertion(self, dfs, config):
        PSMa_accel, PSMb_accel = list(dfs.values())

        n = len(PSMa_accel) if (len(PSMa_accel) < len(PSMb_accel)) else len(PSMb_accel)

        PSMa_accel = PSMa_accel[:n]
        PSMb_accel = PSMb_accel[:n]

        PSMa_accel_magnitude = self.__compute_magnitude(PSMa_accel, config, "fields")
        PSMb_accel_magnitude = self.__compute_magnitude(PSMb_accel, config, "fields")

        PSMa_accel_std = np.std(PSMa_accel_magnitude)
        PSMb_accel_std = np.std(PSMb_accel_magnitude)

        return np.abs ((PSMa_accel_std - PSMb_accel_std)/(PSMa_accel_std + PSMb_accel_std))
    
    def compute_jerk_dispertion(self, dfs, config):
        PSMa_vel, PSMb_vel = list(dfs.values())

        n = len(PSMa_vel) if (len(PSMa_vel) < len(PSMb_vel)) else len(PSMb_vel)

        PSMa_vel = PSMa_vel[:n]
        PSMb_vel = PSMb_vel[:n]

        ts_col = config["timestamp_col"]
        timestamps_a = PSMa_vel[ts_col].values
        timestamps_b = PSMb_vel[ts_col].values

        dt_a = np.diff(timestamps_a)
        dt_b = np.diff(timestamps_b)

        jerk_sqrd_a = np.zeros(n-2)
        jerk_sqrd_b = np.zeros(n-2)

        for field in config["fields"]:
            velocity = PSMa_vel[field]
            acceleration = np.diff(velocity) / dt_a
            dt_accel = dt_a[1:]
            jerk = np.diff(acceleration) / dt_accel
            jerk_sqrd_a += np.square(jerk)
        
        jerk_magnitude_a = np.sqrt(jerk_sqrd_a)

        for field in config["fields"]:
            velocity = PSMb_vel[field]
            acceleration = np.diff(velocity) / dt_b
            dt_accel = dt_b[1:]
            jerk = np.diff(acceleration) / dt_accel
            jerk_sqrd_b += np.square(jerk)
        
        jerk_magnitude_b = np.sqrt(jerk_sqrd_b)

        PSMa_jerk_std = np.std(jerk_magnitude_a)
        PSMb_jerk_std = np.std(jerk_magnitude_b)

        return np.abs( (PSMa_jerk_std - PSMb_jerk_std) / (PSMa_jerk_std + PSMb_jerk_std))
    
    def compute_forcen_magnitude(self, dfs, config):

        trial_data = list(dfs.values())[0]
        # compute speed magnitude at each timestep
        forcen_sq = np.zeros(len(trial_data))
        for field in config["fields"]:
            forcen_sq += trial_data[field] ** 2
        forcen_magnitude = np.sqrt(forcen_sq)
        return float(forcen_magnitude.mean())


    ## nT = normalized by Time 
    ## nPL = normalized by Path Length
    # def compute_acceleration_nT(self, dfs, config):
    #     avg_acceleration_magnitude = self.compute_average_speed_magnitude(dfs, config)
    #     total_time = 

    def compute_metric(self, metric_name, dfs, config):
        if metric_name not in self.metric_fns:
            raise ValueError(f"Metric '{metric_name}' is not supported.")

        compute_fn = self.metric_fns[metric_name]

        if "files" in config:
            return compute_fn(dfs, config)

        results = {}
        for psm_key, psm_config in config.items():
            results[psm_key] = compute_fn(dfs, psm_config)
        return results

    



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

    def load_parquet(self, trial_dir, file_name):
        file_path = os.path.join(trial_dir, file_name)
        if not os.path.exists(file_path):
            return None
        return pd.read_parquet(file_path)

    def load_required_files(self, trial_path, config):
        if "files" in config:
            return {f: self.load_parquet(trial_path, f) for f in config["files"]}
        else:
            return {
                key: {f: self.load_parquet(trial_path, f) for f in subcfg["files"]}
                for key, subcfg in config.items()
            }
        
    def find_subject_files(self, path_to_data):
        return [f for f in os.listdir(path_to_data) if not f.startswith('.')]


    def add_all_subjects(self, manager, path_to_data):
        
        # Grabbing all of the subject directories
        subject_dirs = self.find_subject_files(path_to_data)

        print("Adding all Subjects...")

        # Iterating through each subject
        for subject_dir in subject_dirs:

            # Progress Bar for Preprocess
            pbar = tqdm(desc="Subject " + subject_dir)

            subject_path = os.path.join(path_to_data, subject_dir)

            # Loading all trial directories (directories like T01, T02, etc.)
            subject_trials = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))]

            num_trials = len(subject_trials)
            pbar.total = num_trials

            manager.add_subject(subject_dir, num_trials)

            for subject_trial in subject_trials:

                if os.path.splitext(subject_trial)[1] == ".txt":
                    continue

                manager.add_trial_path(subject_dir, subject_trial)

                pbar.update()

    def compute_metric_for_subject(self, subject_name, metric_name, psms=None):
        if subject_name not in self.subjects:
            raise ValueError(f"No trials found for subject: {subject_name}")

        metrics = self.subjects[subject_name]["metrics"]
        trial_paths = self.subjects[subject_name]["trial_paths"]

        config = self.metric_config[metric_name]

        results = []
        for trial_path in trial_paths:
            if "files" in config:
                dfs = self.load_required_files(trial_path, config)
                if any(df is None for df in dfs.values()):
                    print(f"Skipping {metric_name} for {trial_path}: missing files.")
                    continue
                metric_result = metrics.compute_metric(metric_name, dfs, config)
                results.append({"Trial": trial_path, "Result": metric_result})
            else:
                # multi-PSM config
                trial_results = {}
                for psm_key, psm_config in config.items():
                    if psms is not None and psm_key not in psms:
                        continue
                    dfs = self.load_required_files(trial_path, psm_config)
                    if any(df is None for df in dfs.values()):
                        print(f"Skipping {metric_name} for {trial_path} PSM {psm_key}: missing files.")
                        continue
                    metric_result = metrics.compute_metric(metric_name, dfs, psm_config)
                    trial_results[psm_key] = metric_result
                results.append({"Trial": trial_path, "Result": trial_results})

        return results

    def export_metrics_to_csv(self, metric_names, output_csv_path):
        """
        Export computed metrics for all subjects and trials to a single CSV file.

        Args:
            metric_names (list): List of metric names to compute.
            output_csv_path (str): Path to save the CSV file.
        """
        csv_data = []
        header = ["Subject", "Trial", "Metric", "Domain", "Value"]

        for subject_name, subject_data in self.subjects.items():
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):

                print(subject_name)
                print(trial_path)
                for metric_name in metric_names:
                    config = self.metric_config[metric_name]

                    if "files" in config:
                        dfs = self.load_required_files(trial_path, config)
                        if any(df is None for df in dfs.values()):
                            print(f"Skipping {metric_name} for {trial_path}: missing files.")
                            continue
                        metric_results = metrics.compute_metric(metric_name, dfs, config)

                        # Handle single-value metrics (e.g., completion_time)
                        if isinstance(metric_results, float):
                            row = [subject_name, f"Trial {trial_idx + 1}", metric_name, "N/A", metric_results]
                            csv_data.append(row)
                        # Handle multi-value metrics (e.g., average_speed_magnitude)
                        elif isinstance(metric_results, dict):
                            for domain, value in metric_results.items():
                                row = [subject_name, f"Trial {trial_idx + 1}", metric_name, domain, value]
                                csv_data.append(row)
                        else:
                            raise ValueError(f"Unsupported metric result type for '{metric_name}': {type(metric_results)}")
                    else:
                        # multi-PSM config
                        for psm_key, psm_config in config.items():
                            dfs = self.load_required_files(trial_path, psm_config)
                            if any(df is None for df in dfs.values()):
                                print(f"Skipping {metric_name} for {trial_path} PSM {psm_key}: missing files.")
                                continue
                            metric_results = metrics.compute_metric(metric_name, dfs, psm_config)

                            # For multi-PSM metrics, flatten results by PSM
                            if isinstance(metric_results, float):
                                row = [subject_name, f"Trial {trial_idx + 1}", metric_name, psm_key, metric_results]
                                csv_data.append(row)
                            elif isinstance(metric_results, dict):
                                for domain, value in metric_results.items():
                                    row = [subject_name, f"Trial {trial_idx + 1}", metric_name, f"{psm_key}:{domain}", value]
                                    csv_data.append(row)
                            else:
                                raise ValueError(f"Unsupported metric result type for '{metric_name}' PSM '{psm_key}': {type(metric_results)}")

        df = pd.DataFrame(csv_data, columns=header)
        df.to_csv(output_csv_path, index=False)

        print(f"Metrics saved to {output_csv_path}")

    def export_all_metrics_to_csv(self, metric_names, output_csv_path):
        """
        Export all computed metrics for all subjects and trials to a single CSV file.

        Args:
            metric_names (list): List of metric names to compute.
            output_csv_path (str): Path to save the CSV file.
        """
        csv_data = []
        # For multi-PSM, header should allow for PSM column plus metric columns
        # But since metrics may vary in structure, keep simple header with variable length rows
        # We'll output: Subject, Trial, PSM, Metric, Value for each metric result row

        header = ["Subject", "Trial", "PSM", "Metric", "Value"]

        for subject_name, subject_data in self.subjects.items():
            print(subject_name)
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_idx, trial_path in enumerate(trial_paths):

                print(trial_path)
                for metric_name in metric_names:
                    config = self.metric_config[metric_name]

                    if "files" in config:
                        dfs = self.load_required_files(trial_path, config)
                        if any(df is None for df in dfs.values()):
                            print(f"Skipping {metric_name} for {trial_path}: missing files.")
                            continue
                        metric_result = metrics.compute_metric(metric_name, dfs, config)
                        # Single config metrics have no PSM
                        if isinstance(metric_result, float):
                            row = [subject_name, f"Trial {trial_idx + 1}", "N/A", metric_name, metric_result]
                            csv_data.append(row)
                        elif isinstance(metric_result, dict):
                            for domain, value in metric_result.items():
                                row = [subject_name, f"Trial {trial_idx + 1}", "N/A", f"{metric_name}:{domain}", value]
                                csv_data.append(row)
                        else:
                            raise ValueError(f"Unsupported metric result type for '{metric_name}': {type(metric_result)}")
                    else:
                        # multi-PSM config
                        for psm_key, psm_config in config.items():
                            dfs = self.load_required_files(trial_path, psm_config)
                            if any(df is None for df in dfs.values()):
                                print(f"Skipping {metric_name} for {trial_path} PSM {psm_key}: missing files.")
                                continue
                            metric_result = metrics.compute_metric(metric_name, dfs, psm_config)
                            if isinstance(metric_result, float):
                                row = [subject_name, f"Trial {trial_idx + 1}", psm_key, metric_name, metric_result]
                                csv_data.append(row)
                            elif isinstance(metric_result, dict):
                                for domain, value in metric_result.items():
                                    row = [subject_name, f"Trial {trial_idx + 1}", psm_key, f"{metric_name}:{domain}", value]
                                    csv_data.append(row)
                            else:
                                raise ValueError(f"Unsupported metric result type for '{metric_name}' PSM '{psm_key}': {type(metric_result)}")

        df = pd.DataFrame(csv_data, columns=header)
        df.to_csv(output_csv_path, index=False)

        print(f"All metrics saved to {output_csv_path}")


    def __return_csv_header_ml(self, metric_names):
        csv_data = []
        # For multi-PSM, header should allow for PSM column plus metric columns
        # But since metrics may vary in structure, keep simple header with variable length rows
        # We'll output: Subject, Trial, PSM, Metric, Value for each metric result row

        header = ["Subject_Trial"]

        # --- PATCH FIX START ---
        # Original code only built headers for the *first subject* and *first trial*,
        # then broke out of the loop early.
        # That caused only ~6 trials to export properly.
        #
        # Instead, we now just build the header by iterating over all metric_names directly.
        for metric_name in metric_names:
            config = self.metric_config[metric_name]

            if "files" in config:
                header.append(metric_name)
            else:
                for psm_key, psm_config in config.items():
                    header.append(f"{psm_key}_{metric_name}")
        # --- PATCH FIX END ---

        return header


    def export_metrics_to_csv_ml(self, metric_names, output_csv_path):

        csv_data = []
        header = self.__return_csv_header_ml(metric_names)

        row_data = []
        dataframe = []

        for subject_name, subject_data in self.subjects.items():
            trial_paths = subject_data["trial_paths"]
            metrics = subject_data["metrics"]

            for trial_path in trial_paths:
                print(subject_name)
                print(trial_path)

                trial_id = os.path.basename(trial_path)  # --- PATCH: use real trial ID instead of fake index ---
                row_data.append(f"{subject_name}_{trial_id}")

                for metric_name in metric_names:
                    config = self.metric_config[metric_name]

                    if "files" in config:
                        dfs = self.load_required_files(trial_path, config)
                        if any(df is None for df in dfs.values()):
                            print(f"Skipping {metric_name} for {trial_path}: missing files.")
                            continue
                        metric_results = metrics.compute_metric(metric_name, dfs, config)

                        if isinstance(metric_results, float):
                            row_data.append(metric_results)
                        elif isinstance(metric_results, dict):
                            for domain, value in metric_results.items():
                                row_data.append(value)
                        else:
                            raise ValueError(f"Unsupported metric result type for '{metric_name}': {type(metric_results)}")
                    else:
                        # multi-PSM config
                        for psm_key, psm_config in config.items():
                            dfs = self.load_required_files(trial_path, psm_config)
                            if any(df is None for df in dfs.values()):
                                print(f"Skipping {metric_name} for {trial_path} PSM {psm_key}: missing files.")
                                continue
                            metric_results = metrics.compute_metric(metric_name, dfs, psm_config)

                            if isinstance(metric_results, float):
                                row_data.append(metric_results)
                            elif isinstance(metric_results, dict):
                                for domain, value in metric_results.items():
                                    row_data.append(value)
                            else:
                                raise ValueError(f"Unsupported metric result type for '{metric_name}' PSM '{psm_key}': {type(metric_results)}")
                dataframe.append(row_data)
                row_data = []

        df = pd.DataFrame(dataframe, columns=header)
        df.to_csv(output_csv_path, index=False)

        print(f"Metrics saved to {output_csv_path}")
