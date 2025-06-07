import pandas as pd
import os
import argparse
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt



def find_subject_files(path_to_data):
    return [f for f in os.listdir(path_to_data) if not f.startswith('.')]

def convert_time_to_sec_float(secs, nsecs):

    sec_float = []
    for i in range(len(secs)):
        sec_float.append( secs[i] + (nsecs[i] / 1e9))
    
    return sec_float
        

def chart_data(path_to_data, chosen_ros_topics, path_to_write_data):

    # Establish column header
    column_header = ["Subject", "Trial"] + chosen_ros_topics + ["total rostopics"]
    row_per_trial = []
    data_frame = []

    # Creating directory called data preprocessed in the top level of the repo
    path_to_write_data_ = os.path.join(path_to_write_data, "data_chart")

    if os.path.exists(path_to_write_data_):
        shutil.rmtree(path_to_write_data_)  # Remove directory and all contents

    os.mkdir(path_to_write_data_)

    # Grabbing all of the subject directories
    subject_dirs = find_subject_files(path_to_data)

    print("Preprocessing in Progress...")

    # Iterating through each subject
    for subject_dir in subject_dirs:

        row_per_trial.append(subject_dir)

        # Progress Bar for Preprocess
        pbar = tqdm(desc="Subject " + subject_dir)

        # Loading all trial directories with parquet files
        path_to_parquet_files = os.path.join(path_to_data, subject_dir, "parquet")
        trial_dirs = [d for d in os.listdir(path_to_parquet_files) if not d.startswith('.')]

        pbar.total = len(trial_dirs)

        # Iterating through each trial per subject
        for trial_count, trial_dir in enumerate(trial_dirs):

            if trial_count != 0:
                row_per_trial.append(" ")

            row_per_trial.append(trial_dir)

            num_rostopics_in_trial = 0
            # Iterating through each chosen rostopic to store to preprocessed data folder
            for rostopic_count, rostopic in enumerate(chosen_ros_topics):

                # Load parquet file for rostopic into memory
                parquet_file_path = os.path.join(path_to_parquet_files, trial_dir, rostopic + ".parquet")

                if os.path.exists(parquet_file_path):
                    row_per_trial.append("yes")
                    num_rostopics_in_trial += 1
                else:
                    row_per_trial.append("no")

            row_per_trial.append(num_rostopics_in_trial)
            data_frame.append(row_per_trial)
            row_per_trial = []

            pbar.update()

    # Save the data to a CSV file
    output_csv_path = os.path.join(path_to_write_data_, "Summary.csv")
    df = pd.DataFrame(data_frame, columns=column_header)
    df.to_csv(output_csv_path, index=False)

    print(f"Summary saved to {output_csv_path}")
                
def main():
    print("Starting data preprocessing")

    parser = argparse.ArgumentParser(
                    prog='Preprocess script',
                    description='Processes mistic dataset for surgical robotic skill assessment',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('path_to_data')           # positional argument
    parser.add_argument("path_to_write_data")

    args = parser.parse_args()
    

    # These are the chosen rostopics that we want to preprocess for data collection 
    # CHOSEN_ROS_TOPICS = ["accel_left", "accel_right" ,"consolecamera", "SUJPSM3measured_js", "forcen_left", "MTMR1measured_cp"]
    # CHOSEN_ROS_TOPICS = ["accel_left", "accel_right", "forcen_left"]
    # CHOSEN_ROS_TOPICS = ["accel_left", "accel_right"]

    CHOSEN_ROS_TOPICS = [
    "accel_left",
    "accel_right",
    "ATImini40",
    "consolecamera",
    "consolefollow_mode",
    "consolehead_in",
    "consolehead_out",
    "consoleoperator_present",
    "ECM1bodymeasured_cv",
    "ECM1measured_cp",
    "ECM1measured_js",
    "forcen_left",
    "forcen_right",
    "MTML1bodymeasured_cv",
    "MTML1follow_mode",
    "MTML1grippermeasured_js",
    "MTML1measured_cp",
    "MTML1measured_js",
    "MTML1select",
    "MTMR1bodymeasured_cv",
    "MTMR1follow_mode",
    "MTMR1grippermeasured_js",
    "MTMR1measured_cp",
    "MTMR1measured_js",
    "MTMR1select",
    "PSM1bodymeasured_cv",
    "PSM1follow_mode",
    "PSM1jawmeasured_js",
    "PSM1measured_cp",
    "PSM1measured_js",
    "PSM2bodymeasured_cv",
    "PSM2follow_mode",
    "PSM2jawmeasured_js",
    "PSM2measured_cp",
    "PSM2measured_js",
    "PSM3bodymeasured_cv",
    "PSM3jawmeasured_js",
    "PSM3measured_cp",
    "PSM3measured_js",
    "SUJECM1measured_cp",
    "SUJECM1measured_js",
    "SUJPSM1measured_cp",
    "SUJPSM1measured_js",
    "SUJPSM2measured_cp",
    "SUJPSM2measured_js",
    "SUJPSM3measured_cp",
    "SUJPSM3measured_js"
]

    chart_data (path_to_data=args.path_to_data,     
                    chosen_ros_topics=CHOSEN_ROS_TOPICS,
                    path_to_write_data=args.path_to_write_data)
    



# __name__
if __name__=="__main__":
    main()