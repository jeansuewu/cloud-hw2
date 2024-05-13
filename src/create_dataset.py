from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_dataset(data_path: Path, config: dict) -> pd.DataFrame:
    """Create a structured dataset from raw data.

    Args:
        data_path (Path): Path to the raw data file.
        config (dict): Configuration parameters for creating the dataset.

    Returns:
        pd.DataFrame: The structured dataset.
    """
    # Load the raw data
    try:
        # Attempt to read the raw data from the file
        with open(data_path, 'r') as f:
            data = [[s for s in line.split(' ') if s != '']
                    for line in f.readlines()]
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        return None

    # Get columns from config
    columns = config.get('create_dataset', {}).get('columns', [])

    # Clean the data
    first_cloud = data[53:1077]
    first_cloud = [[float(s.replace('/n', '')) for s in cloud]
                   for cloud in first_cloud]
    first_cloud = pd.DataFrame(first_cloud, columns=columns)
    first_cloud['class'] = np.zeros(len(first_cloud))

    second_cloud = data[1082:2105]
    second_cloud = [[float(s.replace('/n', '')) for s in cloud]
                    for cloud in second_cloud]
    second_cloud = pd.DataFrame(second_cloud, columns=columns)
    second_cloud['class'] = np.ones(len(second_cloud))

    cleaned_data = pd.concat([first_cloud, second_cloud])

    return cleaned_data


def save_dataset(dataset: pd.DataFrame, save_path: Path) -> None:
    """Save the dataset to a CSV file.

    Args:
        dataset (pd.DataFrame): The dataset to save.
        save_path (Path): The path to save the dataset CSV file.
    """
    logger.info("Saving dataset to CSV file...")
    # Save the dataset DataFrame to a CSV file at the specified save_path
    dataset.to_csv(save_path, index=False)
    logger.info("Dataset saved to CSV file at %s", save_path)
