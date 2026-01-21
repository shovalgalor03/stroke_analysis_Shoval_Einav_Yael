import sys
import os
import pytest
import pandas as pd
import numpy as np

# --- Path Setup ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

# Import functions from your file
# Adjust 'src.cluster_analysis' if your file name is different
from src.cluster_analysis import (
    prepare_data, 
    perform_clustering, 
    calculate_cluster_risks,
    get_cluster_profiles,
    find_optimal_k
)

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_prepare_data_logic():
    """
    Logic Check: Verify that data is scaled correctly (Mean approx 0, Std approx 1).
    """
    # 1. Setup
    df = pd.DataFrame({
        'age': [10, 20, 30, 40, 50],
        'bmi': [20, 22, 24, 26, 28],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'] # Categorical to test encoding
    })

    # 2. Run
    X_scaled = prepare_data(df)

    # 3. Assertions
    assert isinstance(X_scaled, np.ndarray)
    # Check shape: 5 rows. Columns should be > 2 because 'gender' gets one-hot encoded
    assert X_scaled.shape[0] == 5 
    assert X_scaled.shape[1] >= 2
    
    # Check Standard Scaling properties (Mean ~ 0, Std ~ 1)
    # We use np.isclose because of floating point tiny errors
    assert np.isclose(X_scaled.mean(), 0, atol=0.1)
    assert np.isclose(X_scaled.std(), 1, atol=0.1)

def test_prepare_data_empty_error():
    """
    Error Handling: Sending empty DataFrame should raise ValueError.
    """
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        prepare_data(df)

def test_perform_clustering_structure():
    """
    Happy Path: Ensure clustering adds the correct columns and no NaNs.
    """
    # 1. Setup
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'stroke': [0, 0, 1, 0, 1, 0]
    })
    # Pre-calculate scaled data (mocking the pipeline flow)
    X_scaled = prepare_data(df)

    # 2. Run (Force k=2)
    result_df = perform_clustering(df, X_scaled, n_clusters=2)

    # 3. Assertions
    assert 'cluster' in result_df.columns
    assert 'group_name' in result_df.columns
    
    # Check that generic names were created
    assert result_df['group_name'].str.contains("Group").all()
    
    # Check for NaNs in cluster assignment
    assert result_df['cluster'].isna().sum() == 0
    
    # Verify shape matches input
    assert len(result_df) == len(df)

def test_calculate_cluster_risks_math():
    """
    Logic Check: Verify the math of stroke risk calculation.
    """
    # 1. Setup: Create a DF where the risk is obvious
    # Group 0: 2 patients, 1 stroke (Risk should be 50%)
    # Group 1: 2 patients, 0 strokes (Risk should be 0%)
    df_clustered = pd.DataFrame({
        'cluster': [0, 0, 1, 1],
        'group_name': ['Group 0', 'Group 0', 'Group 1', 'Group 1'],
        'stroke': [1, 0, 0, 0] 
    })

    # 2. Run
    summary = calculate_cluster_risks(df_clustered)

    # 3. Assertions
    # Check Group 0 (Risk 50%)
    risk_group_0 = summary.loc[summary['cluster'] == 0, 'stroke_risk_%'].values[0]
    assert risk_group_0 == 50.0

    # Check Group 1 (Risk 0%)
    risk_group_1 = summary.loc[summary['cluster'] == 1, 'stroke_risk_%'].values[0]
    assert risk_group_1 == 0.0

def test_get_cluster_profiles_structure():
    """
    Integration: Verify the profile table is generated with correct indices.
    """
    df_clustered = pd.DataFrame({
        'group_name': ['Group 0', 'Group 0', 'Group 1'],
        'age': [20, 30, 40], # Numeric
        'stroke': [0, 1, 0], # Numeric
        'gender': ['M', 'M', 'F'], # Categorical
        'hypertension': [0, 0, 1],
        'heart_disease': [0, 0, 0],
        'avg_glucose_level': [100, 100, 100],
        'bmi': [20, 20, 20],
        'ever_married': ['No', 'No', 'Yes'],
        'work_type': ['A', 'A', 'B'],
        'Residence_type': ['R', 'R', 'U'],
        'smoking_status': ['S', 'S', 'N']
    })

    # Run
    profile_table = get_cluster_profiles(df_clustered)

    # Assertions
    # It should be transposed, so metrics are in the INDEX
    assert 'N (Patients)' in profile_table.index
    assert 'age' in profile_table.index
    assert 'gender' in profile_table.index
    
    # Check columns match the groups
    assert 'Group 0' in profile_table.columns
    assert 'Group 1' in profile_table.columns

def test_find_optimal_k_execution(tmp_path):
    """
    Execution Check: Verify find_optimal_k runs and returns an integer.
    Using tmp_path to save the plot so we don't clutter the disk.
    """
    df = pd.DataFrame(np.random.rand(20, 5), columns=['a','b','c','d','e'])
    X_scaled = prepare_data(df)
    
    # Create a temporary path for the image
    temp_image_path = os.path.join(tmp_path, "test_elbow.png")
    
    # Run
    best_k = find_optimal_k(X_scaled, save_path=temp_image_path, max_k=5)
    
    # Assertions
    assert isinstance(best_k, (int, np.integer))
    assert 1 <= best_k <= 5
    # Check that the file was actually created
    assert os.path.exists(temp_image_path)