import sys
import os
import pytest
import pandas as pd
import numpy as np

# --- Path Setup ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)

from src.load_csv import load_dataset

# -----------------------------------------------------------------------------
# Helper Function
# -----------------------------------------------------------------------------
def create_dummy_csv(filepath, rows=1000, cols=10):
    """
    Helper function to generate a CSV file with specific dimensions.
    """
    data = np.random.rand(rows, cols)
    columns = [f"col_{i}" for i in range(cols)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filepath, index=False)

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_load_dataset_success(tmp_path):
    """
    Positive Test: Verifies that a valid file (1000 rows, 10 cols) loads correctly.
    """
    # 1. Setup: Create a valid CSV file
    file_path = tmp_path / "good_data.csv"
    create_dummy_csv(file_path, rows=1000, cols=10)
    
    # 2. Run function
    result = load_dataset(str(file_path))
    
    # 3. Assertions
    assert result is not None
    assert result.shape == (1000, 10)

def test_load_dataset_insufficient_rows(tmp_path):
    """
    Test failure when the file has fewer than 1000 rows.
    """
    # 1. Setup: Create a file with only 999 rows
    file_path = tmp_path / "low_rows.csv"
    create_dummy_csv(file_path, rows=999, cols=10) 

    # 2. Run function
    result = load_dataset(str(file_path))
    
    # 3. Assertions
    # Function should handle the validation error and return None
    assert result is None

def test_load_dataset_insufficient_columns(tmp_path):
    """
    Test failure when the file has fewer than 10 columns.
    """
    # 1. Setup: Create a file with only 9 columns
    file_path = tmp_path / "low_cols.csv"
    create_dummy_csv(file_path, rows=1000, cols=9) 

    # 2. Run function
    result = load_dataset(str(file_path))

    # 3. Assertions
    # Function should handle the validation error and return None
    assert result is None

def test_file_not_found():
    """
    Negative Test: Verifies behavior when file does not exist.
    """
    # 1. Setup: Define a non-existent file path
    fake_path = "ghost_file.csv"

    # 2. Run function
    result = load_dataset(fake_path)
    
    # 3. Assertions
    # Underlying error: FileNotFoundError -> Returns None
    assert result is None

def test_empty_file(tmp_path):
    """
    Negative Test: Verifies behavior when file exists but is empty.
    """
    # 1. Setup: Create an empty file (0 bytes)
    file_path = tmp_path / "empty.csv"
    file_path.touch() 

    # 2. Run function
    result = load_dataset(str(file_path))
    
    # 3. Assertions
    # Underlying error: EmptyDataError -> Returns None
    assert result is None

def test_corrupted_csv(tmp_path):
    """
    Negative Test: Verifies behavior when file contains malformed data.
    """
    # 1. Setup: Write invalid CSV format (unclosed quote)
    file_path = tmp_path / "broken.csv"
    with open(file_path, "w") as f:
        f.write('col1,col2\n"value_without_closing_quote') 

    # 2. Run function
    result = load_dataset(str(file_path))
    
    # 3. Assertions
    # Underlying error: ParserError -> Returns None
    assert result is None