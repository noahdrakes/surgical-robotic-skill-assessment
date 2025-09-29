import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_force_sensor_z_from_parquet(parquet_file):
    """
    Plots the Z-axis AT40mini force sensor data directly from a Parquet file.

    Args:
        parquet_file (str): Path to the Parquet file containing raw data.
    """
    # Check if the file exists
    if not os.path.exists(parquet_file):
        print(f"Error: Parquet file not found at {parquet_file}")
        return

    # Load the Parquet file
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    # Combine seconds and nanoseconds for high-resolution timestamp
    if 'header_stamp_nsec' in df.columns:
        df['timestamp'] = df['header_stamp_sec'] + df['header_stamp_nsec'] / 1e9
    else:
        df['timestamp'] = df['header_stamp_sec']

    # Find the Z-axis force column
    force_cols = [c for c in df.columns if c.endswith('_force_z') or c == 'force_z']
    if not force_cols:
        print("Error: Z-axis force field not found in the Parquet file.")
        return
    force_col = force_cols[0]

    # Plot the Z-axis force field
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df[force_col], label=f"{force_col} (Z-Axis)")

    # Add labels and legend
    plt.title("AT40mini Force Sensor Z-Axis Data (Raw Parquet)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Force Z (N)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Plot AT40mini Z-axis force sensor data directly from a Parquet file.")
    parser.add_argument("parquet_file", help="Path to the Parquet file containing raw data.")
    args = parser.parse_args()

    # Call the plot function
    plot_force_sensor_z_from_parquet(args.parquet_file)