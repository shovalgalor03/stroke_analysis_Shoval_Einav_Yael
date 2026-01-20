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
# NOTE: Update 'src.chi_square_analysis' to the actual filename where you saved the function!
# Example: from src.hypothesis_testing import create_contingency_table
from src.chi_square_analysis import create_contingency_table 

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_contingency_success():
    """
    Happy Path: Validate that a 2x2 contingency table is created correctly.
    """
    # 1. Setup: Valid data with 2 categories in each column
    df = pd.DataFrame({
        'gender': ['Male', 'Male', 'Female', 'Female'],
        'stroke': ['Yes', 'No', 'Yes', 'No']
    })

    # 2. Run function
    result = create_contingency_table(df, 'gender', 'stroke')

    # 3. Assertions
    assert result is not None, "Function returned None but expected a DataFrame."
    assert isinstance(result, pd.DataFrame)
    
    # Check dimensions (should be 2x2)
    assert result.shape == (2, 2)
    
    # Check specific counts (Male/Yes = 1, Female/No = 1, etc.)
    # Note: Access using .loc[Row, Col]
    assert result.loc['Male', 'Yes'] == 1
    assert result.loc['Female', 'No'] == 1

def test_contingency_handles_nans():
    """
    Data Cleaning: Verify that rows with NaNs are dropped, 
    but the table is still created if enough data remains.
    """
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', np.nan], # One missing value
        'outcome': ['Yes', 'No', 'Yes', 'No', 'Yes']
    })

    result = create_contingency_table(df, 'group', 'outcome')

    assert result is not None
    # Total count should be 4 (5 rows minus 1 NaN)
    assert result.sum().sum() == 4
    assert result.shape == (2, 2)

def test_contingency_missing_columns():
    """
    Error Handling: One of the requested columns does not exist.
    Should catch KeyError and return None.
    """
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    # 'C' does not exist
    result = create_contingency_table(df, 'A', 'C')
    
    assert result is None

def test_contingency_too_small():
    """
    Edge Case: Data exists, but doesn't have enough categories for a 2x2 table.
    (e.g., 'Gender' column only has 'Female').
    Should catch AssertionError and return None.
    """
    df = pd.DataFrame({
        'gender': ['Female', 'Female', 'Female'], # Only 1 category
        'stroke': ['Yes', 'No', 'Yes']          # 2 categories
    })
    
    # Resulting table would be (1, 2), which violates the 2x2 rule
    result = create_contingency_table(df, 'gender', 'stroke')
    
    assert result is None

def test_contingency_empty_dataframe():
    """
    Error Handling: Input DataFrame is empty.
    Should catch ValueError/Exception and return None.
    """
    df = pd.DataFrame()
    
    result = create_contingency_table(df, 'col1', 'col2')
    
    assert result is None

def test_contingency_invalid_input_type():
    """
    Error Handling: Input is not a DataFrame.
    Should catch TypeError and return None.
    """
    not_df = {"key": "value"}
    
    result = create_contingency_table(not_df, 'key', 'value') # type: ignore
    
    assert result is None