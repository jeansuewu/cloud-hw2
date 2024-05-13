import numpy as np


def generate_additional_features(data):
    """
    Generates additional features based on the existing ones.

    Args:
    - data (pd.DataFrame): The original dataset.

    Returns:
    - features (pd.DataFrame): The dataset with additional features.
    """
    columns = ['visible_mean', 'visible_max', 'visible_min',
               'visible_mean_distribution', 'visible_contrast', 
               'visible_entropy', 'visible_second_angular_momentum', 
               'IR_mean', 'IR_max', 'IR_min']

    features = data[columns]
    target = data['class']
    
    features['log_entropy'] = features.visible_entropy.apply(np.log)
    # Check for division by zero error
    if 0 in features['visible_entropy'].values:
        raise ValueError("Division by zero error: visible_entropy contains zero values.")
    features['entropy_x_contrast'] = features.visible_contrast.multiply(
        features.visible_entropy)
    # Check if IR_min greater than IR_max to avoid division by zero error
    if any(data['IR_min'] == data['IR_max']):
        raise ValueError("Division by zero error: IR_min greater than IR_max.")
    features['IR_range'] = features.IR_max - features.IR_min
    # Check if IR_mean is zero
    if (features['IR_mean'] == 0).any():
        # Set IR_norm_range to NaN if IR_mean is zero
        features['IR_norm_range'] = np.nan
    else:
        # Calculate IR_norm_range normally if IR_mean is not zero
        features['IR_norm_range'] = (features.IR_max - features.IR_min) / features.IR_mean

    return target, features
