import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_force_sensor_z(parquet_folder, trial_name):
    """
    Plots the Z-axis AT40mini force sensor data for a single trial.

    Args:
        parquet_folder (str): Path to the folder containing Parquet files.
        trial_name (str): Name of the trial to plot (e.g., "Trial1").
    """
    # Path to the AT40mini Parquet file for the given trial
    parquet_file_path = os.path.join(parquet_folder, trial_name, "ATImini40.parquet")

    # Check if the file exists
    if not os.path.exists(parquet_file_path):
        print(f"Error: Parquet file not found at {parquet_file_path}")
        return

    # Load the Parquet file
    df = pd.read_parquet(parquet_file_path, engine="pyarrow")

    # Check if the Z-axis force field exists
    if "force_z" not in df.columns:
        print("Error: Z-axis force field ('force_z') not found in the Parquet file.")
        return

    # Plot the Z-axis force field
    plt.figure(figsize=(10, 6))
    plt.plot(df["header_stamp_sec"], df["force_z"], label="Force Z-Axis")

    # Add labels and legend
    plt.title(f"AT40mini Force Sensor Z-Axis Data - {trial_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Force Z (N)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Plot AT40mini Z-axis force sensor data for a single trial.")
    parser.add_argument("parquet_folder", help="Path to the folder containing Parquet files.")
    parser.add_argument("trial_name", help="Name of the trial to plot (e.g., 'Trial1').")
    args = parser.parse_args()

    # Call the plot function
    plot_force_sensor_z(args.parquet_folder, args.trial_name)