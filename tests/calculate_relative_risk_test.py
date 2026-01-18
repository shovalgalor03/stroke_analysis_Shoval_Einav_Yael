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
from src.relative_risk_analysis import calculate_relative_risk

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_calculate_rr_success():
    """
    Happy Path: Validates that the function correctly processes a DataFrame
    and returns a dictionary with the expected results.
    """
    # 1. Setup: Create a simple DataFrame
    # Group A: 1 sick, 1 healthy (50% risk)
    # Group B: 0 sick, 2 healthy (0% risk)
    df = pd.DataFrame({
        'risk_group': ['GroupA', 'GroupA', 'GroupB', 'GroupB'],
        'stroke':     [1,        0,        0,        0]
    })

    # 2. Run function
    result = calculate_relative_risk(
        df, 
        exposed_group='GroupA', 
        control_group='GroupB',
        outcome_col='stroke',
        group_col='risk_group'
    )

    # 3. Assertions
    assert result is not None, "Function returned None but valid result was expected."
    assert isinstance(result, dict), "Result should be a dictionary."
    
    # Check specific keys
    assert result['comparison'] == "GroupA vs GroupB"
    assert result['Exposed_Cases'] == 1
    assert result['Control_Cases'] == 0
    # RR should be inf (0.5 / 0), handled gracefully by the inner logic
    assert result['RR'] == np.inf 

def test_calculate_rr_missing_group():
    """
    Edge Case: One of the requested groups does not exist in the DataFrame.
    Should log a warning and return None (not crash).
    """
    df = pd.DataFrame({
        'risk_group': ['GroupA', 'GroupB'],
        'stroke': [0, 1]
    })

    # 'GroupZ' does not exist
    result = calculate_relative_risk(df, 'GroupA', 'GroupZ')

    # Expecting None because the group wasn't found
    assert result is None

def test_calculate_rr_missing_column():
    """
    Edge Case: The specified outcome column is missing.
    The function catches ValueError and returns None.
    """
    df = pd.DataFrame({
        'risk_group': ['A', 'B'],
        'stroke': [0, 1]
    })

    # typo in column name ('heart_attack' instead of 'stroke')
    result = calculate_relative_risk(df, 'A', 'B', outcome_col='heart_attack')

    assert result is None

def test_calculate_rr_invalid_outcome_type():
    """
    Edge Case: Outcome column contains strings instead of numbers.
    The function checks for numeric type and raises TypeError -> returns None.
    """
    df = pd.DataFrame({
        'risk_group': ['A', 'B'],
        'stroke': ['yes', 'no'] # Invalid! Should be 0/1
    })

    result = calculate_relative_risk(df, 'A', 'B')

    assert result is None

def test_calculate_rr_not_a_dataframe():
    """
    Safety Check: Input is a list, not a DataFrame.
    Should return None.
    """
    not_df = [{'risk_group': 'A', 'stroke': 1}]
    
    result = calculate_relative_risk(not_df, 'A', 'B')
    
    assert result is None