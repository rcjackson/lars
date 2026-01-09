import pandas as pd

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