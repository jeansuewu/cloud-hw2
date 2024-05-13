import pytest
import pandas as pd
import numpy as np
import pytest

from src.generate_features import generate_additional_features

@pytest.fixture
def sample_data():
    # Sample data for testing
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [15, 25, 35],
        'visible_min': [5, 15, 25],
        'visible_mean_distribution': [0.1, 0.2, 0.3],
        'visible_contrast': [10, 20, 30],
        'visible_entropy': [0.5, 0.6, 0.7],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 60, 70],
        'IR_max': [55, 65, 75],
        'IR_min': [45, 55, 65],
        'class': [0, 1, 0]  # Sample target values
    })
    return data

def test_log_entropy_feature_happy_path(sample_data):
    # Happy path test for the log_entropy feature
    target, features = generate_additional_features(sample_data)
    expected_log_entropy = np.log(sample_data['visible_entropy'])
    np.testing.assert_allclose(features['log_entropy'], expected_log_entropy)

def test_log_entropy_feature_unhappy_path(sample_data):
    # Unhappy path test for the log_entropy feature
    sample_data['visible_entropy'] = 0  # Cause division by zero error
    with pytest.raises(ValueError):
        target, features = generate_additional_features(sample_data)

def test_entropy_x_contrast_feature_happy_path(sample_data):
    # Happy path test for the entropy_x_contrast feature
    target, features = generate_additional_features(sample_data)
    expected_entropy_x_contrast = sample_data['visible_contrast'] * sample_data['visible_entropy']
    np.testing.assert_allclose(features['entropy_x_contrast'], expected_entropy_x_contrast)

# def test_entropy_x_contrast_feature_unhappy_path(sample_data):
#     # Unhappy path test for the entropy_x_contrast feature
    sample_data_copy = sample_data.copy()  # Make a copy to avoid modifying the original data
    sample_data_copy['visible_entropy'] = 0  # Set visible_entropy to zero
    with pytest.raises(ValueError):
        target, features = generate_additional_features(sample_data_copy)

def test_IR_range_feature_happy_path(sample_data):
    # Happy path test for the IR_range feature
    target, features = generate_additional_features(sample_data)
    expected_IR_range = sample_data['IR_max'] - sample_data['IR_min']
    np.testing.assert_allclose(features['IR_range'], expected_IR_range)

def test_IR_range_feature_unhappy_path(sample_data):
    # Unhappy path test for the IR_range feature where IR_min equals IR_max
    sample_data_copy = sample_data.copy()  # Make a copy to avoid modifying the original data
    sample_data_copy['IR_max'] = sample_data_copy['IR_min']  # Set IR_max equal to IR_min
    with pytest.raises(ValueError):
        target, features = generate_additional_features(sample_data_copy)

def test_IR_norm_range_feature_happy_path(sample_data):
    # Happy path test for the IR_norm_range feature
    target, features = generate_additional_features(sample_data)
    expected_IR_norm_range = (sample_data['IR_max'] - sample_data['IR_min']) / sample_data['IR_mean']
    np.testing.assert_allclose(features['IR_norm_range'], expected_IR_norm_range)

def test_IR_norm_range_feature_unhappy_path(sample_data):
    # Unhappy path test for the IR_norm_range feature where IR_mean is zero
    sample_data_copy = sample_data.copy()  # Make a copy to avoid modifying the original data
    sample_data_copy['IR_mean'] = 0  # Set IR_mean to zero
    target, features = generate_additional_features(sample_data_copy)
    
    # Check if any value in IR_norm_range is NaN
    assert features['IR_norm_range'].isnull().any()
