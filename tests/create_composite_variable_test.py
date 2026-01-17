import sys
import os
import pytest
import pandas as pd

# --- Path Setup (Crucial for importing from src) ---
# This block ensures Python can find the 'src' directory
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import after setting up the path
from src.transformers import create_composite_variable

def test_create_composite_logic():
    """
    Check that risk groups are assigned correctly based on flags.
    """
    # 1. Create data covering all combinations
    df = pd.DataFrame({
        'bmi_high':     [0, 1, 0, 1],
        'glucose_high': [0, 0, 1, 1]
    })
    # Row 0: 0/0 -> neither
    # Row 1: 1/0 -> bmi_only
    # Row 2: 0/1 -> glucose_only
    # Row 3: 1/1 -> both_high
    
    # 2. Run function
    result = create_composite_variable(df)
    
    # 3. Assertions and comparisons
    expected_groups = ['neither', 'bmi_only', 'glucose_only', 'both_high']
    assert result['risk_group'].tolist() == expected_groups

def test_create_composite_invalid_values():
    """
    Test error when flags contain invalid values (not 0/1).
    Checks for a general Exception using pytest.raises.
    """
    # 1. Create bad data (invalid flag value)
    df_bad = pd.DataFrame({'bmi_high': [5], 'glucose_high': [0]})
    
    # 2. Expect ANY exception (Logic preserved as requested)
    pytest.raises(Exception, create_composite_variable, df_bad)