import sys
import os
import pytest
import pandas as pd
import numpy as np

# --- Path Setup ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import the pipeline function
# (Make sure this function is in src/analysis_pipeline.py or similar)
# Adjust the import based on your actual file name!
from src.relative_risk_analysis import run_full_analysis_pipeline 

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_pipeline_full_success():
    """
    Happy Path: Validates that the pipeline runs for ALL groups 
    and returns a summary DataFrame with 3 rows.
    """
    # 1. Setup: Create data containing ALL groups
    # We need: neither, bmi_only, glucose_only, both_high
    data = {
        'risk_group': [
            'neither', 'neither',      # Control
            'bmi_only', 'bmi_only',    # Group 1
            'glucose_only', 'glucose_only', # Group 2
            'both_high', 'both_high'   # Group 3
        ],
        'stroke': [
            0, 0,  # neither (0% risk)
            1, 0,  # bmi (50% risk)
            1, 0,  # glucose (50% risk)
            1, 1   # both (100% risk)
        ]
    }
    df = pd.DataFrame(data)

    # 2. Run the full pipeline
    results_df = run_full_analysis_pipeline(df)

    # 3. Assertions
    assert not results_df.empty, "Result DataFrame should not be empty"
    # We expect 3 comparisons (bmi, glucose, both)
    assert len(results_df) == 3
    
    # Check that crucial columns exist
    assert 'comparison' in results_df.columns
    assert 'RR' in results_df.columns
    assert 'P_Value' in results_df.columns

    # Verify logic: 'both_high' should be in the comparisons
    comparisons = results_df['comparison'].tolist()
    assert any("both_high" in c for c in comparisons)

def test_pipeline_partial_data():
    """
    Edge Case: Data is missing for some groups (e.g., 'glucose_only' is missing).
    The pipeline should skip the missing group but calculate the others.
    """
    # 1. Setup: Missing 'glucose_only'
    data = {
        'risk_group': ['neither', 'neither', 'both_high', 'both_high'],
        'stroke':     [0,        0,         1,           0]
    }
    df = pd.DataFrame(data)

    # 2. Run pipeline
    results_df = run_full_analysis_pipeline(df)

    # 3. Assertions
    # We expect only 1 comparison result (both_high vs neither)
    # 'bmi_only' and 'glucose_only' are missing from input
    assert len(results_df) == 1
    assert "both_high" in results_df.iloc[0]['comparison']

def test_pipeline_empty_input():
    """
    Edge Case: Input DataFrame is None or Empty.
    Should return an empty DataFrame immediately without crashing.
    """
    # Test with None
    res1 = run_full_analysis_pipeline(None)
    assert isinstance(res1, pd.DataFrame)
    assert res1.empty

    # Test with Empty DataFrame
    res2 = run_full_analysis_pipeline(pd.DataFrame())
    assert isinstance(res2, pd.DataFrame)
    assert res2.empty