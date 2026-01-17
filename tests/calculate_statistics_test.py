import pytest
import numpy as np
from src.relative_risk_analysis import calculate_statistics

def calculate_statistics_test():
    """
    Test standard case:
    Exposed: 10/100 sick (10%)
    Control: 5/100 sick (5%)
    Expected RR = 2.0
    """
    # 1. Run the function with known inputs
    # Exposed group: 10 sick, 90 healthy (Total 100, Risk = 0.1)
    # Control group: 5 sick, 95 healthy (Total 100, Risk = 0.05)
    rr, ci_lower, ci_upper, p_val = calculate_statistics(10, 90, 5, 95)
    
    # 2. Assertions
    
    # Verify that Relative Risk (RR) is calculated correctly
    # 0.1 / 0.05 should be exactly 2.0
    assert rr == 2.0
    
    # Verify that Confidence Intervals are valid numbers (not NaN)
    assert not np.isnan(ci_lower)
    assert not np.isnan(ci_upper)
    
    # Verify that P-value is a valid probability (between 0 and 1)
    assert 0 <= p_val <= 1

def test_calculate_statistics_zero_control_risk():
    """Test infinite risk when control group has 0 sick cases."""
    
    # Control group has 0 sick people, meaning 0 risk.
    # Mathematically, dividing by zero leads to Infinity.
    # We expect numpy to return np.inf (Infinity).
    rr, _, _, _ = calculate_statistics(10, 90, 0, 100)
    
    assert rr == np.inf

def test_calculate_statistics_empty_groups():
    """Test that empty groups return NaN."""
    
    # Sending 0 participants for all groups.
    # Risk calculation is impossible (0 divided by 0).
    rr, _, _, _ = calculate_statistics(0, 0, 0, 0)
    
    # We expect the result to be NaN (Not a Number) to handle this gracefully
    # rather than crashing.
    assert np.isnan(rr)