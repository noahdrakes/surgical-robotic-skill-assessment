import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Hardcoded ANOVA p-values ===
data = {
    "Metric": [
        "completion_time",
        "PSM1_average_speed_magnitude",
        "PSM2_average_speed_magnitude",
        "PSM1_speed_effort_nT",
        "PSM2_speed_effort_nT",
        "PSM1_speed_effort_nPL",
        "PSM2_speed_effort_nPL",
        "PSM1_average_acceleration_magnitude",
        "PSM2_average_acceleration_magnitude",
        "PSM1_acceleration_effort_nPL",
        "PSM2_acceleration_effort_nPL",
        "PSM1_acceleration_effort_nT",
        "PSM2_acceleration_effort_nT",
        "PSM1_average_jerk_magnitude",
        "PSM2_average_jerk_magnitude",
        "PSM1_jerk_effort_nT",
        "PSM2_jerk_effort_nT",
        "PSM1_jerk_effort_nPL",
        "PSM2_jerk_effort_nPL",
        "ATIForceSensor_average_force_magnitude",
        "ATIForceSensor_average_force_magnitude_nT",
        "ATIForceSensor_average_force_magnitude_nPL",
        "ATIForceSensor_average_force_magnitude_1",
        "PSM1_total_path_length",
        "PSM2_total_path_length",
        "PSM1_average_angular_speed_magnitude",
        "PSM2_average_angular_speed_magnitude",
        "PSM1_PSM2_speed_correlation",
        "PSM1_PSM2_speed_cross",
        "PSM1_PSM2_acceleration_cross",
        "PSM1_PSM2_jerk_cross",
        "PSM1_PSM2_acceleration_dispertion",
        "PSM1_PSM2_jerk_dispertion",
        "ATIForceSensor_max_force_magnitude",
        "ATIForceSensor_max_force_magnitude_nT",
        "ATIForceSensor_max_force_magnitude_nPL",
        "ATIForceSensor_min_force_magnitude",
        "ATIForceSensor_max_force_x",
        "ATIForceSensor_max_force_y",
        "ATIForceSensor_max_force_z",
    ],
    "p_value": [
        7.32074906349897e-05,
        5.69039519034246e-15,
        5.17475550813082e-14,
        5.551353737471e-15,
        5.39672794937172e-14,
        0.000366740456563943,
        0.203943866427185,
        0.335145588162976,
        0.817528322514475,
        2.61381493726203e-05,
        7.76464583193057e-05,
        0.334971047090619,
        0.817728672467443,
        3.68599558439604e-07,
        0.000214030096964736,
        3.28746127811805e-07,
        0.000264568756924229,
        5.38478459538515e-09,
        2.58988556932328e-08,
        0.307543193249689,
        2.24440584312499e-07,
        0.00678455218803797,
        0.307543193249689,
        0.0309527026183139,
        0.0367074804802552,
        2.93078098063412e-06,
        1.3024617242617e-05,
        0.0220108805453705,
        0.215228279195828,
        0.0137268866850694,
        0.390077774254485,
        0.000354023512276501,
        0.506955515997169,
        0.865557743662807,
        0.0048102658276668,
        0.384116709584922,
        0.0758922681972281,
        0.706232285760533,
        0.136054944145697,
        0.898753484392337,
    ],
}

# === Convert to DataFrame and sort ===
df = pd.DataFrame(data)
df = df.sort_values("p_value", ascending=True).reset_index(drop=True)

# === Plot ===
plt.figure(figsize=(10, max(6, len(df) * 0.25)))
bars = plt.barh(df["Metric"], df["p_value"], color="#2196f3", alpha=0.85)
plt.xscale("log")  # log scale to visualize small values
plt.axvline(0.05, color="red", linestyle="--", label="p = 0.05 significance threshold")
plt.xlabel("p-value (log scale)")
plt.ylabel("Metric")
plt.title("ANOVA p-values by Metric (sorted)")
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("anova_pvalues_sorted.png", dpi=300)
plt.show()

print("✅ Saved plot as 'anova_pvalues_sorted.png'")