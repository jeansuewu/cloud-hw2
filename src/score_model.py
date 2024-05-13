import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

def score_model(test_data, model):
    """
    Scores a trained model on the test data.

    Args:
    - test_data (pd.DataFrame): The testing data containing features and target variable.
    - model: The trained model.

    Returns:
    - scores (dict): A dictionary containing evaluation scores.
    """
    # Extract features and target variable from test data
    x_test = test_data.drop(columns=['class'])
    y_test = test_data['class']

    # Predict target variable using the trained model
    y_pred = model.predict(x_test)

    # Calculate evaluation scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create dictionary to store evaluation scores
    scores = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return scores


def save_scores(scores, filepath):
    """
    Saves evaluation scores to a CSV file.

    Args:
    - scores (dict): A dictionary containing evaluation scores.
    - filepath (str or Path): The file path where scores will be saved.
    """
    # Convert scores dictionary to DataFrame
    scores_df = pd.DataFrame(scores, index=[0])

    # Save scores DataFrame to CSV file
    scores_df.to_csv(filepath, index=False)

    logger.info("Evaluation scores saved to CSV file.")
