import pandas as pd

def is_trial_valid(trial_inclusion_matrix_csv, Subject, trial):

    df = pd.read_csv(trial_inclusion_matrix_csv)
    subject_num = int(str(Subject[1] + str(Subject[2])))

    print(subject_num)

    trial_status = df.loc[subject_num - 1, trial]

    if trial_status == 'P':
        return True
    else:
        return False


# file = "/Users/noahdrakes/Documents/research/skill_assessment/MISTIC_robotic_suturing_study/protocol/trial_inclusion_matrix.csv"

# df = pd.read_csv("/Users/noahdrakes/Documents/research/skill_assessment/MISTIC_robotic_suturing_study/protocol/trial_inclusion_matrix.csv")
# # print(df)
# print(df.loc[2,"T01"])

# print(is_trial_valid(file, "S02", "T08"))


# sub = "S01"
# num = str(sub[1]) + str(sub[2])

# num = int(num)
# print(num)