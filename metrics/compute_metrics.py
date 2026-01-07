import os
import pandas as pd
import math
import argparse
import shutil
from pprint import pprint
from Metrics import Metrics, MetricsManager



## printing completion time
manager = MetricsManager(config_path="metric_config.json")

print("adding all subjects")

path_to_data = "/home/ndrakes1/surgical_skill_assessment/surgical-robotic-skill-assessment/preprocess/preprocessed_data"
# path_to_data = "/home/ndrakes1/surgical_skill_assessment/surgical-robotic-skill-assessment/preprocess/preprocessed_data_no_bad_instruments"
manager.add_all_subjects(manager, path_to_data)

print(manager.subjects)
# exit()

print("subjects added")

output_csv_path = "results.csv"

print("exporting to csv")

# metric_names = ["completion_time", "average_speed_magnitude", "average_acceleration_magnitude", "average_jerk_magnitude", "jerk_effort_nT", "jerk_effort_nPL",
#                 "average_force_magnitude", "total_path_length", "average_angular_speed_magnitude", "speed_correlation", "speed_cross",
#                 "acceleration_cross", "jerk_cross", "acceleration_dispertion", "jerk_dispertion", "forcen_magnitude", "forcen_cross", "forcen_correlation", "forcen_dispertion",
#                 "max_force_magnitude", "min_force_magnitude", "max_force_x", "max_force_y", ]

metric_names = ["completion_time", 
                "average_speed_magnitude", 
                "average_acceleration_magnitude", 
                "acceleration_effort_nPL",
                "acceleration_effort_nT",
                "average_jerk_magnitude", 
                "jerk_effort_nT",
                "jerk_effort_nPL",
                "average_force_magnitude",
                "average_force_magnitude_nT",
                "average_force_magnitude_nPL",
                "average_force_magnitude", 
                "total_path_length", 
                "average_angular_speed_magnitude", 
                "speed_correlation", 
                "speed_cross",
                "acceleration_cross", 
                "jerk_cross", 
                "acceleration_dispertion", 
                "jerk_dispertion",
                "max_force_magnitude", 
                "max_force_magnitude_nT",
                "max_force_magnitude_nPL",
                "min_force_magnitude", 
                "max_force_x", 
                "max_force_y",
                "max_force_z" ]

# metric_names = ["completion_time", 
#                 "average_speed_magnitude", 
#                 "average_acceleration_magnitude", 
#                 "acceleration_effort_nPL",
#                 "acceleration_effort_nT",
#                 "average_jerk_magnitude", 
#                 "jerk_effort_nT",
#                 "jerk_effort_nPL",
#                 "average_force_magnitude",
#                 "average_force_magnitude_nT",
#                 "average_force_magnitude_nPL",
#                 "average_force_magnitude", 
#                 "total_path_length", 
#                 "average_angular_speed_magnitude", 
#                 "speed_correlation", 
#                 "speed_cross",
#                 "acceleration_cross", 
#                 "jerk_cross", 
#                 "acceleration_dispertion", 
#                 "jerk_dispertion",
#                 "max_force_magnitude", 
#                 "max_force_magnitude_nT",
#                 "max_force_magnitude_nPL",
#                 "min_force_magnitude", 
#                 "max_force_x", 
#                 "max_force_y",
#                 "max_force_z",
#                 "forcen_cross",
#                 "forcen_magnitude",
#                 "forcen_dispertion",
#                 "forcen_correlation" ]

print()
print()
print("ALL METRICS: ", metric_names)

# manager.return_csv_header_ml(metric_names)
# exit()
manager.export_metrics_to_csv_ml(metric_names, "results_ml.csv")
manager.export_metrics_to_csv(metric_names, "results.csv")
print("finished")