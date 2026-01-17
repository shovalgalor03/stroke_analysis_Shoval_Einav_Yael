import sys
import os
import pytest
import numpy as np

# --- Path Setup (Crucial) ---
# This fixes the "ModuleNotFoundError" by helping Python find the 'src' folder
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import from the correct file
from src.relative_risk_analysis import calculate_statistics

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_calculate_statistics_standard():
    """
    Test standard case (Happy Path):
    Exposed: 10/100 sick (10%)
    Control: 5/100 sick (5%)
    Expected RR = 2.0
    """
    # 1. Run the function with known inputs
    # a=10 (sick exposed), b=90 (healthy exposed) -> Total 100
    # c=5 (sick control), d=95 (healthy control) -> Total 100
    rr, ci_lower, ci_upper, p_val = calculate_statistics(10, 90, 5, 95)
    
    # 2. Assertions
    # Verify Relative Risk calculation: 0.10 / 0.05 = 2.0
    assert rr == 2.0
    
    # Verify CI values are valid (not NaN and logically ordered)
    assert not np.isnan(ci_lower)
    assert not np.isnan(ci_upper)
    assert ci_lower < rr < ci_upper
    
    # Verify P-value is a valid probability
    assert 0 <= p_val <= 1

def test_calculate_statistics_zero_control_risk():
    """
    Edge Case: Infinite risk when control group has 0 sick cases.
    """
    # Control group has 0 sick people (c=0).
    # Risk Control = 0.
    # RR = Risk Exposed / 0 -> Infinity.
    rr, _, _, _ = calculate_statistics(10, 90, 0, 100)
    
    assert rr == np.inf

def test_calculate_statistics_zero_exposed_risk():
    """
    Edge Case: Zero risk when exposed group has 0 sick cases.
    """
    # Exposed group has 0 sick people (a=0).
    # RR = 0 / Risk Control -> 0.0
    rr, _, _, _ = calculate_statistics(0, 100, 5, 95)
    
    assert rr == 0.0

def test_calculate_statistics_identical_groups():
    """
    Edge Case: Both groups have identical risk.
    RR should be exactly 1.0.
    """
    # 10/100 vs 10/100
    rr, _, _, _ = calculate_statistics(10, 90, 10, 90)
    
    assert rr == 1.0

def test_calculate_statistics_empty_groups():
    """
    Error Handling: Test that empty groups (zeros) return NaN safely.
    """
    # Sending 0 participants for all groups.
    # Risk calculation involves division by zero (0/0).
    rr, ci_lower, ci_upper, p_val = calculate_statistics(0, 0, 0, 0)
    
    # We expect the result to be NaN (Not a Number)
    assert np.isnan(rr)
    assert np.isnan(ci_lower)
    assert np.isnan(ci_upper)