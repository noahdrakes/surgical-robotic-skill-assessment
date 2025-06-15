import pandas as pd
import os
import argparse
import shutil
from tqdm import tqdm
import numpy as np



def find_subject_files(path_to_data):
    return [f for f in os.listdir(path_to_data) if not f.startswith('.')]

def convert_time_to_sec_float(secs, nsecs):

    sec_float = []
    for i in range(len(secs)):
        sec_float.append( secs[i] + (nsecs[i] / 1e9))
    
    return sec_float

def calculate_sample_rate(timestamps):
    timestamps_ = np.array(timestamps)
    return 1 / np.diff(timestamps_).mean(0)
    

def preprocess_data(trial_dir, chosen_ros_topics):

    new_df = pd.DataFrame()
    # iterating through each chosen rostopic to store to preprocessed ata folder
    for rostopic_count, rostopic in enumerate(chosen_ros_topics):

        # load parquet file for rostopic into memory
        parquet_file_path = os.path.join(trial_dir, rostopic + ".parquet")
        df = pd.read_parquet(parquet_file_path, "pyarrow")

        float_sec = []

        # we are grabbing the timestamp from the first rostopic 
        sec_column = df['header_stamp_sec']
        nsec_column = df['header_stamp_nsec']
        float_sec = convert_time_to_sec_float(sec_column, nsec_column)

        sample_rate = calculate_sample_rate(float_sec)
        
        print("ROSTOPIC: ", rostopic, " | sample_rate: ", sample_rate, "Hz ")



                


def main():
    print("Starting data preprocessing")

    parser = argparse.ArgumentParser(
                    prog='Preprocess script',
                    description='Processes mistic dataset for surgical robotic skill assessment',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('path_to_data')           # positional argument

    args = parser.parse_args()
    

    # These are the chosen rostopics that we want to preprocess for data collection 
    # CHOSEN_ROS_TOPICS = ["accel_left", "accel_right" ,"consolecamera", "SUJPSM3measured_js", "forcen_left", "MTMR1measured_cp"]
    # CHOSEN_ROS_TOPICS = ["accel_left", "accel_right", "forcen_left"]
    # CHOSEN_ROS_TOPICS = ["PSM1measured_cp", "PSM2measured_cp", "ATImini40"]

    CHOSEN_ROS_TOPICS = [
    "accel_left",
    "accel_right",
    "ATImini40",
    "consolecamera",
    "consolefollow_mode",
    # "consolehead_in",
    # "consolehead_out",
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

    preprocess_data(trial_dir=args.path_to_data,     
                    chosen_ros_topics=CHOSEN_ROS_TOPICS)
    



# __name__
if __name__=="__main__":
    main()