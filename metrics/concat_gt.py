import csv
import pandas as pd


def concatenate_labels(path_to_labels, path_to_metrics):

    labels_df = pd.read_csv(path_to_labels)
    subjects = list(labels_df["SUBJECT"])
    # map each array element to its index
    subject_lookup = {val: i for i, val in enumerate(subjects)}

    metrics_df = pd.read_csv(path_to_metrics)
    metrics_df["LABEL"] = None


    labels_df = labels_df.set_index("SUBJECT")

    for i in range(len(metrics_df)):

        chosen_subject = ""
        for subject in subjects:
            if subject in metrics_df.loc[i, "Subject_Trial"]:
                chosen_subject = subject
                break
        
        chosen_label = labels_df.loc[chosen_subject, "LABEL"]
        metrics_df.loc[i, "LABEL"] = chosen_label

    print(metrics_df)
    metrics_df.to_csv(path_to_metrics, index=False)


concatenate_labels("/Users/noahdrakes/Documents/research/skill_assessment/surgical-robotic-skill-assessment/labels/labels.csv","/Users/noahdrakes/Documents/research/skill_assessment/surgical-robotic-skill-assessment/metrics/results.csv")








