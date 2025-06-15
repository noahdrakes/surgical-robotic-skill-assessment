import pandas as pd
import os
import argparse
import shutil
from tqdm import tqdm
from list_sample_rate_freq import calculate_sample_rate
from downsample_freq import downsample_ros_data



def find_subject_files(path_to_data):
    return [f for f in os.listdir(path_to_data) if not f.startswith('.')]

def convert_time_to_sec_float(secs, nsecs):

    sec_float = []
    for i in range(len(secs)):
        sec_float.append( secs[i] + (nsecs[i] / 1e9))
    
    return sec_float


def preprocess_data(path_to_data, chosen_ros_topics, path_to_write_data):

    # all rostopics 
    rostopics_header = []

    # Creating directory called data preprocessed in the top level of the repo
    path_to_write_data_ = os.path.join(path_to_write_data, "data_preprocessed")

    # print("is this the error")
    if os.path.exists(path_to_write_data_):
        # os.rmdir(path_to_write_data_)
        # remove dir and all of its contents
        shutil.rmtree(path_to_write_data_)

    
    os.mkdir(path_to_write_data_)
    
    # grabbing all of the subject dir
    subject_dirs = find_subject_files(path_to_data)

    print("Preprocessing in Progress...")

    # iterating through each subject
    for subject_dir in subject_dirs:

        # Progress Bar for Preprocess
        pbar = tqdm(desc="Subject " + subject_dir)

        # loading all trial dir with parquet files
        path_to_parquet_files = os.path.join(path_to_data, subject_dir, "parquet")
        trial_dirs = [d for d in os.listdir(path_to_parquet_files) if not d.startswith('.')]

        pbar.total = len(trial_dirs)

        # iterating through each trial per subject
        for trial_count, trial_dir in enumerate(trial_dirs):

            new_df = pd.DataFrame()
            # iterating through each chosen rostopic to store to preprocessed ata folder
            for rostopic_count, rostopic in enumerate(chosen_ros_topics):

                # load parquet file for rostopic into memory
                parquet_file_path = os.path.join(path_to_parquet_files, trial_dir, rostopic + ".parquet")
                df = pd.read_parquet(parquet_file_path, "pyarrow")

                df = downsample_ros_data(df, target_frequency = "40.00ms")

                float_sec = []

                # we are grabbing the timestamp from the first rostopic 
                if rostopic_count == 0:
                    sec_column = df['header_stamp_sec']
                    nsec_column = df['header_stamp_nsec']
                    float_sec = convert_time_to_sec_float(sec_column, nsec_column)
                    new_df = pd.DataFrame({'timestamp': float_sec})

                    # adding timestamp to header
                    if trial_count == 0:
                        if "timestamp" not in rostopics_header:
                            rostopics_header.append("timestamp")

                # HARDCODED start and end idx -> may need to change depending on if the parquet data format changes
                parquet_start_idx = 4
                parquet_end_index = len(df.columns) - 1 # this is the column before the bag name

                # iterated through each element in the rostopic 
                for i in range(parquet_start_idx, parquet_end_index):

                    column_name = df.columns[i]
                    column_data = df[column_name]  # This gives you the whole column as a Series

                    # delete angular accel accleration
                    if "angular_accel" in column_name:
                        continue

                    rostopics_header_ = rostopic + "_" + column_name

                    new_df[rostopics_header_] = column_data # adding rostopic name to each header

                    if trial_count == 0:
                        if rostopics_header_ not in rostopics_header:
                            rostopics_header.append(rostopics_header_)

                ## FOR DEBUGGING ##
                # print(rostopic)
                # pd.set_option('display.max_columns', None)
                # print(df.head())
            

            subject_dir_ = os.path.join(path_to_write_data_, subject_dir)
            os.makedirs(subject_dir_, exist_ok=True)

            trial_path = os.path.join(path_to_write_data_, subject_dir, trial_dir + ".csv")
            new_df.to_csv(trial_path, index=False, header=False)

            pbar.update()
        
        rostopics_header_path = os.path.join(path_to_write_data_, subject_dir, "rostopics_header.txt")
        with open(rostopics_header_path, "w") as f:
            f.write(",".join(rostopics_header) + "\n")


                


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
    CHOSEN_ROS_TOPICS = ["PSM1measured_cp", "PSM2measured_cp", "ATImini40"]

#     CHOSEN_ROS_TOPICS = [
#     "accel_left",
#     "accel_right",
#     "ATImini40",
#     "consolecamera",
#     "consolefollow_mode",
#     "consolehead_in",
#     "consolehead_out",
#     "consoleoperator_present",
#     "ECM1bodymeasured_cv",
#     "ECM1measured_cp",
#     "ECM1measured_js",
#     "forcen_left",
#     "forcen_right",
#     "MTML1bodymeasured_cv",
#     "MTML1follow_mode",
#     "MTML1grippermeasured_js",
#     "MTML1measured_cp",
#     "MTML1measured_js",
#     "MTML1select",
#     "MTMR1bodymeasured_cv",
#     "MTMR1follow_mode",
#     "MTMR1grippermeasured_js",
#     "MTMR1measured_cp",
#     "MTMR1measured_js",
#     "MTMR1select",
#     "PSM1bodymeasured_cv",
#     "PSM1follow_mode",
#     "PSM1jawmeasured_js",
#     "PSM1measured_cp",
#     "PSM1measured_js",
#     "PSM2bodymeasured_cv",
#     "PSM2follow_mode",
#     "PSM2jawmeasured_js",
#     "PSM2measured_cp",
#     "PSM2measured_js",
#     "PSM3bodymeasured_cv",
#     "PSM3jawmeasured_js",
#     "PSM3measured_cp",
#     "PSM3measured_js",
#     "SUJECM1measured_cp",
#     "SUJECM1measured_js",
#     "SUJPSM1measured_cp",
#     "SUJPSM1measured_js",
#     "SUJPSM2measured_cp",
#     "SUJPSM2measured_js",
#     "SUJPSM3measured_cp",
#     "SUJPSM3measured_js"
# ]

    preprocess_data(path_to_data=args.path_to_data,     
                    chosen_ros_topics=CHOSEN_ROS_TOPICS,
                    path_to_write_data=args.path_to_write_data)
    



# __name__
if __name__=="__main__":
    main()