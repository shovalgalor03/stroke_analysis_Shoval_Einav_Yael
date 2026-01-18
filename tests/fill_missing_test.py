import sys
import os
import pytest
import pandas as pd
import numpy as np

# --- Path Setup (Crucial) ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import the function from the correct file
# (Assuming it is in src/data_cleaning.py like the previous function)
from src.data_cleaning import fill_missing_with_median

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_fill_median_success():
    """
    Happy Path: Verify that NaNs are filled with the correct median value.
    """
    # 1. Setup: Data with missing values
    # Values: 10, 20, 30. Median is 20.
    df = pd.DataFrame({
        'age': [10, 20, np.nan, 30] 
    })

    # 2. Run function
    result = fill_missing_with_median(df, 'age')

    # 3. Assertions
    # Verify no missing values remain
    assert result['age'].isna().sum() == 0
    
    # Verify the NaN was replaced by 20.0
    # The original NaN was at index 2
    assert result['age'].iloc[2] == 20.0

def test_fill_median_no_missing_values():
    """
    Edge Case: Column has no missing values.
    The function should return the dataframe unchanged.
    """
    df = pd.DataFrame({
        'age': [10, 20, 30]
    })

    result = fill_missing_with_median(df, 'age')

    # Values should match exactly
    pd.testing.assert_frame_equal(df, result)

def test_fill_median_non_numeric_error():
    """
    Error Handling: Input column is not numeric (e.g., strings).
    The function catches the AssertionError and returns the original DF.
    """
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', np.nan]
    })

    # Run function on a string column
    result = fill_missing_with_median(df, 'name')

    # Assertions
    # The function should FAIL to fill (because of type check) and return original
    assert result['name'].isna().sum() == 1
    assert result['name'].iloc[2] is np.nan

def test_fill_median_all_nans():
    """
    Edge Case: Column contains ONLY NaNs.
    Median cannot be calculated (it is NaN).
    The function should catch this and return original DF without crashing.
    """
    df = pd.DataFrame({
        'age': [np.nan, np.nan, np.nan]
    })

    result = fill_missing_with_median(df, 'age')

    # Should remain all NaNs because median calculation failed
    assert result['age'].isna().sum() == 3

def test_fill_median_missing_column():
    """
    Error Handling: The requested column does not exist.
    """
    df = pd.DataFrame({'age': [10, 20]})
    
    # Ask for 'height' which doesn't exist
    result = fill_missing_with_median(df, 'height')
    
    # Should return original dataframe safely
    assert 'height' not in result.columns
    pd.testing.assert_frame_equal(df, result)