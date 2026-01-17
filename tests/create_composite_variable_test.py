import pytest
import pandas as pd
from src.transformers import create_composite_variable

def test_create_composite_logic():
    """Check that risk groups are assigned correctly based on flags."""
    df = pd.DataFrame({
        'bmi_high':     [0, 1, 0, 1],
        'glucose_high': [0, 0, 1, 1]
    })
    
    result = create_composite_variable(df)
    
    expected_groups = ['neither', 'bmi_only', 'glucose_only', 'both_high']
    assert result['risk_group'].tolist() == expected_groups

def test_create_composite_invalid_values():
    """Test error when flags contain invalid values (not 0/1)."""
    df_bad = pd.DataFrame({'bmi_high': [5], 'glucose_high': [0]})
    
    # We expect ANY exception
    pytest.raises(Exception, create_composite_variable, df_bad)