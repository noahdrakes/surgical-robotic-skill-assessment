import pandas as pd
import os
import argparse
import shutil
from tqdm import tqdm



def find_subject_files(path_to_data):
    return [f for f in os.listdir(path_to_data) if not f.startswith('.')]

def convert_time_to_sec_float(secs, nsecs):

    sec_float = []
    for i in range(len(secs)):
        sec_float.append( secs[i] + (nsecs[i] / 1e9))
    
    return sec_float
        

def list_elements(path_to_data, chosen_ros_topics):


    print("Preprocessing in Progress...")

    # iterating through each chosen rostopic to store to preprocessed ata folder
    for rostopic_count, rostopic in enumerate(chosen_ros_topics):

        # load parquet file for rostopic into memory
        parquet_file_path = os.path.join(path_to_data, rostopic + ".parquet")
        df = pd.read_parquet(parquet_file_path, "pyarrow")

        shape = []

        if rostopic_count == 0:
            shape = df.shape
        

        print("")
        print("ROSTOPIC: ", rostopic, " shape: ", df.shape)
        print("sub elements")
        print("____________")

        # HARDCODED start and end idx -> may need to change depending on if the parquet data format changes
        parquet_start_idx = 4
        parquet_end_index = len(df.columns) - 1 # this is the column before the bag name

        # iterated through each element in the rostopic 
        for i in range(parquet_start_idx, parquet_end_index):

            column_name = df.columns[i]
            print(column_name)

            # delete angular accel accleration
            if "angular_accel" in column_name:
                continue

def main():
    print("hello")
    print("Starting data preprocessing")

    parser = argparse.ArgumentParser(
                    prog='Preprocess script',
                    description='Processes mistic dataset for surgical robotic skill assessment',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('path_to_data', help='path to the first trial with all of the parquet files')           # positional argument

    args = parser.parse_args()
    
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

    list_elements(path_to_data=args.path_to_data, chosen_ros_topics=CHOSEN_ROS_TOPICS)
    



# __name__
if __name__=="__main__":
    main()