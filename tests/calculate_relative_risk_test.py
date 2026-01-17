import pandas as pd
from src.relative_risk_analysis import calculate_relative_risk

def calculate_relative_risk_test():
    """Test RR calculation logic with a DataFrame."""
    df = pd.DataFrame({
        'risk_group': ['both_high']*100 + ['neither']*100,
        'stroke':     [1]*20 + [0]*80 +    # both_high: 20% risk
                      [1]*5 + [0]*95       # neither: 5% risk
    })
    
    result = calculate_relative_risk(df, 'both_high', 'neither')
    
    assert result is not None
    assert result['comparison'] == 'both_high vs neither'
    assert 3.5 < result['RR'] < 4.5
    assert result['Significant_0.05'] is True

def test_calculate_rr_missing_group():
    """Test that non-existent groups return None."""
    df = pd.DataFrame({'risk_group': ['A'], 'stroke': [0]})
    result = calculate_relative_risk(df, 'Ghost', 'A')
    assert result is None