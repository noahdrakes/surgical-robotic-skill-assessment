import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_force_sensor_z(csv_file, rostopic_header_file):
    """
    Plots the Z-axis AT40mini force sensor data from preprocessed CSV data.

    Args:
        csv_file (str): Path to the preprocessed CSV file (no header).
        rostopic_header_file (str): Path to the rostopic header file containing column names.
    """
    # Load rostopic header to get column names
    if not os.path.exists(rostopic_header_file):
        print(f"Error: Rostopic header file not found at {rostopic_header_file}")
        return

    with open(rostopic_header_file, "r") as f:
        column_names = f.readline().strip().split(",")

    # Load the CSV file with the column names
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return

    data = pd.read_csv(csv_file, header=None)
    data.columns = column_names

    # Check if the Z-axis force field exists
    if "ATImini40_force_z" not in data.columns:
        print("Error: Z-axis force field ('ATImini40_force_z') not found in the CSV file.")
        return

    # Plot the Z-axis force field
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["ATImini40_force_z"], label="Force Z-Axis")

    # Add labels and legend
    plt.title("AT40mini Force Sensor Z-Axis Data")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Force Z (N)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Plot AT40mini Z-axis force sensor data from preprocessed CSV data.")
    parser.add_argument("csv_file", help="Path to the preprocessed CSV file (no header).")
    parser.add_argument("rostopic_header_file", help="Path to the rostopic header file containing column names.")
    args = parser.parse_args()

    # Call the plot function
    plot_force_sensor_z(args.csv_file, args.rostopic_header_file)