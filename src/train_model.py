from pathlib import Path
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def train_model(target, features, test_size, n_estimators, max_depth):
    """
    Trains a Random Forest classifier on the training data.

    Args:
    - target (pd.DataFrame): The features for training.
    - features (pd.Series): The target variable for training.
    - test_size (float): The proportion of the dataset to include in the test split.
    - n_estimators (int): The number of trees in the forest.
    - max_depth (int): The maximum depth of the trees.

    Returns:
    - model: The trained Random Forest classifier.
    - train (pd.DataFrame): The combined DataFrame of features and target for training.
    - test (pd.DataFrame): The combined DataFrame of features and target for testing.
    """
    logger.info("Training started...")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth)
    model.fit(x_train, y_train)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    logger.info("Training completed.")

    return model, train, test


def save_data(train, test, save_path: Path) -> None:
    """Save the train and test datasets to disk.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The testing dataset.
        save_path (Path): The path to save the datasets.
    """
    logger.info("Saving the train and test data")
    try:
        train.to_csv(save_path / "train.csv", index=False)
        test.to_csv(save_path / "test.csv", index=False)
    except FileNotFoundError as err:
        logger.error(
            "Please provide a valid path to store the train and test data: %s", err
        )

    logger.info("Data saved successfully. Train and test data saved to %s", save_path)


def save_model(model, save_path: Path) -> None:
    """Save the trained model to disk.

    Args:
        model: The trained model.
        save_path (Path): The path to save the trained model.
    """
    logger.info("Saving the trained model into a pickle file")
    try:
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
    except FileNotFoundError as error:
        logger.error("File path not found. Please provide a \
                     valid path to store the model: %s", error)

    logger.info("Finished saving trained model to %s", save_path)
