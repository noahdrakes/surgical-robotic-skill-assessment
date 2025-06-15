from list_sample_rate_freq import calculate_sample_rate
import pandas as pd

def downsample_ros_data(df, target_frequency="20.833ms"):
    """
    Downsamples a ROS DataFrame based on the header timestamp.

    Args:
        df (pd.DataFrame): The DataFrame containing ROS data with header fields.
        target_frequency (str): Target frequency for resampling (e.g., '20.833ms').

    Returns:
        pd.DataFrame: Downsampled DataFrame.
    """
    # Combine header_stamp_sec and header_stamp_nsec into a single timestamp column
    df["timestamp"] = df["header_stamp_sec"] + df["header_stamp_nsec"] / 1e9

    # Convert the timestamp column to a datetime format for resampling
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)

    # Select only numeric columns for resampling
    numeric_columns = df.select_dtypes(include=["number"]).columns
    downsampled_df = df[numeric_columns].resample(target_frequency).mean()

    # Handle missing values (fill or interpolate)
    downsampled_df = downsampled_df.interpolate(method="linear").ffill()

    # Reset the index to return a DataFrame with the original structure
    downsampled_df.reset_index(inplace=True)

    return downsampled_df