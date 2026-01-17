import sys
import os
import pytest
import pandas as pd
import numpy as np

# --- Path Setup (Crucial for importing from src) ---
# This block ensures Python can find the 'src' directory
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import the function to be tested
# (Assuming the function is located in src/data_cleaning.py)
from src.data_cleaning import convert_to_numeric

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_convert_to_numeric():
    """
    Happy Path: Verify that mixed strings/numbers are correctly converted to float.
    'N/A' should become NaN without triggering the safety threshold.
    """
    # 1. Setup: Data with convertible strings and one missing value
    df = pd.DataFrame({
        'salary': ['1000', '2000.5', 'N/A', '3000'] 
    })
    # 25% data loss (1 out of 4) is acceptable (default threshold is 0.5 or 50%)

    # 2. Run function
    result = convert_to_numeric(df, 'salary')

    # 3. Assertions
    # Check if column is now numeric (float)
    assert pd.api.types.is_numeric_dtype(result['salary'])
    
    # Check values
    assert result['salary'].iloc[0] == 1000.0
    assert result['salary'].iloc[1] == 2000.5
    assert pd.isna(result['salary'].iloc[2]) # The 'N/A' should be NaN

def test_convert_numeric_safety_threshold_exceeded():
    """
    Safety Check: Test that conversion ABORTS if too much data is lost.
    Input has 75% garbage data, Default Threshold is 50%.
    """
    # 1. Setup: Data with mostly non-numeric values
    df = pd.DataFrame({
        'salary': ['1000', 'invalid', 'error', 'text']
    })
    # 3 out of 4 are bad -> 75% loss. This should trigger the assertion error.

    # 2. Run function
    # Note: The function catches the error internally and returns the ORIGINAL df.
    result = convert_to_numeric(df, 'salary', safety_threshold=0.5)

    # 3. Assertions
    # The column should remain as Object (String), NOT converted to numeric
    assert not pd.api.types.is_numeric_dtype(result['salary'])
    assert result['salary'].dtype == 'object'
    
    # Values should be untouched
    assert result['salary'].iloc[1] == 'invalid'

def test_convert_numeric_custom_threshold():
    """
    Edge Case: Verify that changing the safety_threshold works.
    We allow 80% data loss, so the conversion should succeed despite bad data.
    """
    # 1. Setup: High data loss scenario
    df = pd.DataFrame({
        'salary': ['1000', 'bad', 'bad', 'bad'] # 75% loss
    })

    # 2. Run function with loose threshold (0.8 or 80%)
    result = convert_to_numeric(df, 'salary', safety_threshold=0.8)

    # 3. Assertions
    # Should succeed because 0.75 < 0.8
    assert pd.api.types.is_numeric_dtype(result['salary'])
    assert pd.isna(result['salary'].iloc[1])

def test_convert_numeric_missing_column():
    """
    Error Handling: Test behavior when the column does not exist.
    """
    # 1. Setup
    df = pd.DataFrame({'age': [20, 30]})

    # 2. Run function on non-existent column
    result = convert_to_numeric(df, 'ghost_column')

    # 3. Assertions
    # Should return original dataframe gracefully (no crash)
    assert 'ghost_column' not in result.columns
    assert result.shape == (2, 1)

def test_convert_numeric_already_numeric():
    """
    Edge Case: The column is already numeric. Nothing should change.
    """
    # 1. Setup
    df = pd.DataFrame({'age': [20.5, 30.1, 40.0]})

    # 2. Run function
    result = convert_to_numeric(df, 'age')

    # 3. Assertions
    assert pd.api.types.is_numeric_dtype(result['age'])
    assert result['age'].iloc[0] == 20.5