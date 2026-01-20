import sys
import os
import pytest
import pandas as pd
import numpy as np

# --- Path Setup (Crucial) ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import the function
# Update 'src.outliers' if the file name is different
from src.outliers import remove_outliers_iqr

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_remove_outliers_success():
    """
    Happy Path: Create a dataset with obvious outliers and verify removal.
    """
    # 1. Setup: 
    # Logic: Median is approx 5. IQR is small. 100 is definitely an outlier.
    data = {'value': [4, 5, 5, 6, 5, 100]} 
    df = pd.DataFrame(data)

    # 2. Run function
    result = remove_outliers_iqr(df, 'value', threshold=1.5)

    # 3. Assertions
    # We expect the '100' to be removed, leaving 5 rows.
    assert len(result) == 5
    assert 100 not in result['value'].values
    assert result['value'].max() <= 6

def test_remove_outliers_no_outliers():
    """
    Happy Path: Dataset has no outliers. 
    The function should return the dataframe unchanged (or copy of it).
    """
    df = pd.DataFrame({'value': [10, 11, 12, 10, 11]})
    
    result = remove_outliers_iqr(df, 'value')
    
    # Assertions
    assert len(result) == len(df)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

def test_remove_outliers_helper_failure():
    """
    Integration/Edge Case: The helper function 'calculate_iqr_bounds' fails 
    (e.g., due to missing column) and returns None.
    The main function should catch this and return the original DF.
    """
    df = pd.DataFrame({'value': [1, 2, 3]})
    
    # Requesting a column that does NOT exist ('ghost')
    # This usually causes the helper to return None (or log error)
    result = remove_outliers_iqr(df, 'ghost')
    
    # Expect original dataframe back
    pd.testing.assert_frame_equal(result, df)

def test_remove_outliers_type_mismatch():
    """
    Error Handling: Column exists but contains strings (cannot calculate IQR).
    The function should catch the error and return original DF.
    """
    df = pd.DataFrame({'value': ['a', 'b', 'c', 'd']})
    
    result = remove_outliers_iqr(df, 'value')
    
    # Expect original dataframe back (safely aborted)
    pd.testing.assert_frame_equal(result, df)

def test_remove_outliers_high_count_warning():
    """
    Edge Case: Verify that the function still works even if it logs a warning
    about removing too many outliers (>10%).
    """
    # 1. Setup: 
    # Create 100 rows of '1' (Normal data) and 20 rows of '5000' (Outliers).
    # This guarantees that Q1 and Q3 will both be 1, so IQR = 0.
    # Any value > 1 will be removed.
    
    normal_data = [1] * 100
    outliers = [5000] * 20
    
    df = pd.DataFrame({'value': normal_data + outliers})

    # 2. Run function
    result = remove_outliers_iqr(df, 'value')

    # 3. Assertions
    # We expect only the 100 normal values to remain
    assert len(result) == 100
    
    # Verify that the high values are gone
    assert result['value'].max() == 1