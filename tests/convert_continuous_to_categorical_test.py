import pytest
import pandas as pd
from src.transformers import convert_continuous_to_categorical

def convert_continuous_to_categorical_test():
    """
    Test convert_continuous_to_categorical covering all scenarios.
    Assumed Thresholds: BMI >= 30, Glucose >= 140.
    """
    # 1. Create data covering all combinations
    df = pd.DataFrame({
        'bmi': [20, 35, 35, 20],               
        'avg_glucose_level': [100, 200, 100, 200] 
    })
    # Row 0: Low/Low -> neither
    # Row 1: High/High -> both_high
    # Row 2: High/Low -> bmi_only
    # Row 3: Low/High -> glucose_only

    # 2. Run function
    result = convert_continuous_to_categorical(df)

    # 3. Assertions and comparisons 
    assert result['bmi_high'].tolist() == [0, 1, 1, 0]
    assert result['glucose_high'].tolist() == [0, 1, 0, 1]
    
    expected_groups = ['neither', 'both_high', 'bmi_only', 'glucose_only']
    assert result['risk_group'].tolist() == expected_groups

def test_convert_missing_columns():
    """
    Test failure when required columns are missing.
    Checks for a general Exception.
    """
    df_bad = pd.DataFrame({'bmi': [22]})
    
    # We expect ANY exception
    pytest.raises(Exception, convert_continuous_to_categorical, df_bad)