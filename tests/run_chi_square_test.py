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
# Update 'src.chi_square_analysis' if the file name is different
from src.chi_square_analysis import run_chi_square_test

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_chi_square_significant_relation():
    """
    Happy Path: Test a scenario with a very strong correlation.
    Group A is almost always 'Sick', Group B is almost always 'Healthy'.
    Expect: Significant Result (p < 0.05).
    """
    # 1. Setup: Strong pattern
    # Group A: 50 Sick, 0 Healthy
    # Group B: 0 Sick, 50 Healthy
    data = {
        'group':   ['A'] * 50 + ['B'] * 50,
        'outcome': ['Sick'] * 50 + ['Healthy'] * 50
    }
    df = pd.DataFrame(data)

    # 2. Run function
    result = run_chi_square_test(df, 'group', 'outcome')

    # 3. Assertions
    assert result is not None
    assert result['significant'] == True
    assert result['p_value'] < 0.05
    assert result['interpretation'] == "Significant (Reject H0)"

def test_chi_square_no_relation():
    """
    Happy Path: Test a scenario with NO correlation (Random distribution).
    Expect: Not Significant (p > 0.05).
    """
    # 1. Setup: Balanced/Random data
    # Both groups have 50/50 split of Yes/No
    data = {
        'group':   ['A', 'A'] * 50 + ['B', 'B'] * 50,
        'outcome': ['Yes', 'No'] * 50 + ['Yes', 'No'] * 50
    }
    df = pd.DataFrame(data)

    # 2. Run function
    result = run_chi_square_test(df, 'group', 'outcome')

    # 3. Assertions
    assert result is not None
    assert result['significant'] == False
    assert result['p_value'] > 0.05
    assert "Fail to Reject H0" in result['interpretation']

def test_chi_square_data_prep_failure():
    """
    Integration: Test that if 'create_contingency_table' fails (returns None),
    this function also returns None gracefully.
    """
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    # Passing a non-existent column 'Z' causing data prep to fail
    result = run_chi_square_test(df, 'A', 'Z')
    
    assert result is None

def test_chi_square_invalid_alpha_type():
    """
    Error Handling: Passing an invalid type for alpha (string instead of float).
    Should be caught by the TypeError block.
    """
    df = pd.DataFrame({
        'group': ['A', 'B'],
        'outcome': ['Yes', 'No']
    })
    
    # Passing alpha as a string "0.05" triggers TypeError when comparing p_val < alpha
    result = run_chi_square_test(df, 'group', 'outcome', alpha="0.05") # type: ignore
    
    assert result is None

def test_chi_square_small_sample_warning():
    """
    Edge Case: Small sample size.
    The test runs, logs a warning about 'expected frequencies < 5', 
    but should still return a valid result dictionary.
    """
    df = pd.DataFrame({
        'group': ['A', 'B', 'A'],
        'outcome': ['Yes', 'No', 'No']
    })
    
    # 2x2 table with very small counts
    result = run_chi_square_test(df, 'group', 'outcome')
    
    # Should still succeed and calculate numbers
    assert result is not None
    assert isinstance(result['p_value'], float)
    assert isinstance(result['chi2_statistic'], float)