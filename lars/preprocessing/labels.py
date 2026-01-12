import pandas as pd
import os

def change_file_path(radar_df, new_path):
    """
    Change the file paths in the radar DataFrame to a new path.

    Parameters
    ----------
    radar_df (pd.DataFrame): DataFrame containing radar data with file paths.
    new_path (str): New base path to replace in the file paths.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated file paths.
    """
    radar_df = radar_df.copy()
    radar_df['file_path'] = radar_df['file_path'].apply(
        lambda x: os.path.join(new_path, os.path.basename(x))
    )
    return radar_df

def load_labels(label_file):
    """
    Load labels from a CSV file.

    Parameters
    ----------
    label_file (str): Path to the CSV file containing labels.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the labels.
    """
    return pd.read_csv(label_file)

def save_labels(label_df, output_file):
    """
    Save labels to a CSV file.

    Parameters
    ----------
    label_df (pd.DataFrame): DataFrame containing the labels.
    output_file (str): Path to save the CSV file.
    """
    label_df.to_csv(output_file, index=False)