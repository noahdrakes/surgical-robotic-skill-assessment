import pandas as pd
import argparse
import os

def parquet_to_csv(parquet_file, output_csv):
    """
    Converts a Parquet file to a CSV file.

    Args:
        parquet_file (str): Path to the input Parquet file.
        output_csv (str): Path to save the output CSV file.
    """
    # Check if the Parquet file exists
    if not os.path.exists(parquet_file):
        print(f"Error: Parquet file '{parquet_file}' does not exist.")
        return

    # Load the Parquet file
    try:
        df = pd.read_parquet(parquet_file)
        print(f"Successfully loaded Parquet file: {parquet_file}")
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    # Save the DataFrame to a CSV file
    try:
        df.to_csv(output_csv, index=False)
        print(f"Successfully saved CSV file: {output_csv}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    # Argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Convert a Parquet file to a CSV file.")
    parser.add_argument("parquet_file", help="Path to the input Parquet file.")
    parser.add_argument("output_csv", help="Path to save the output CSV file.")
    args = parser.parse_args()

    # Convert Parquet to CSV
    parquet_to_csv(args.parquet_file, args.output_csv)