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
# Change 'src.data_cleaning' if your function is in a different file
from src.data_cleaning import remove_columns

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_remove_columns_success():
    """
    Happy Path: Verify that specified columns are actually removed.
    """
    # 1. Setup: DataFrame with 3 columns
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'age': [25, 30, 35],
        'name': ['A', 'B', 'C']
    })

    # 2. Run function: Remove 'id' and 'name'
    result = remove_columns(df, ['id', 'name'])

    # 3. Assertions
    # Columns 'id' and 'name' should be gone
    assert 'id' not in result.columns
    assert 'name' not in result.columns
    # Column 'age' should remain
    assert 'age' in result.columns
    # Shape should be (3, 1)
    assert result.shape == (3, 1)

def test_remove_columns_missing_cols():
    """
    Edge Case: Try to remove a column that does not exist.
    The function should warn (log) but NOT crash, and remove the others.
    """
    df = pd.DataFrame({
        'age': [25, 30],
        'gender': ['F', 'M']
    })

    # 'ghost_col' does not exist, 'gender' does.
    result = remove_columns(df, ['ghost_col', 'gender'])

    # 3. Assertions
    # 'gender' should be removed
    assert 'gender' not in result.columns
    # 'age' should stay
    assert 'age' in result.columns
    # The function handled the missing column gracefully
    assert result.shape == (2, 1)

def test_remove_columns_empty_list():
    """
    Edge Case: The list of columns to remove is empty.
    Should return the DataFrame unchanged.
    """
    df = pd.DataFrame({'age': [10, 20]})
    
    result = remove_columns(df, [])
    
    # Using testing.assert_frame_equal to ensure exact match
    pd.testing.assert_frame_equal(df, result)

def test_remove_columns_invalid_input_type():
    """
    Error Handling: Input 'columns_to_remove' is not a list (e.g., a string).
    The function catches TypeError and returns the original object.
    """
    df = pd.DataFrame({'age': [10, 20]})
    
    # Mistake: passing a string 'age' instead of list ['age']
    result = remove_columns(df, 'age') # type: ignore
    
    # Logic in function catches TypeError and returns 'df'
    # So the column is NOT removed because the input was invalid
    assert 'age' in result.columns
    pd.testing.assert_frame_equal(df, result)

def test_remove_columns_not_a_dataframe():
    """
    Error Handling: The input 'df' is not a DataFrame.
    Should catch TypeError and return the input as is.
    """
    not_df = {"key": "value"}
    
    result = remove_columns(not_df, ['key']) # type: ignore
    
    # Should return the dictionary back
    assert result == not_df

def test_remove_columns_case_mismatch():
    """
    Edge Case: Case sensitivity (e.g., trying to remove 'Age' instead of 'age').
    The function logic separates this into 'missing_cols' and suggests a fix,
    but it does NOT remove the column (pandas is case-sensitive).
    """
    df = pd.DataFrame({'age': [10, 20]})
    
    # Try to remove 'Age' (Capitalized)
    result = remove_columns(df, ['Age'])
    
    # 'age' (lowercase) should still be there because 'Age' wasn't found
    assert 'age' in result.columns   