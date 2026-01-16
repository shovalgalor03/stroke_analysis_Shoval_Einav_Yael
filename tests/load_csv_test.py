import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# --- Path Setup ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)
# ------------------

try:
    from src.load_csv import load_dataset
except ImportError:
    raise ImportError("Could not find 'load_dataset'. Make sure 'load_csv.py' is inside the 'src' folder.")

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    """Mocks the logger within src.load_csv."""
    with patch("src.load_csv.logger") as mock:
        yield mock

@pytest.fixture
def valid_dataframe():
    """Creates a valid DataFrame (1000 rows, 10 cols)."""
    data = np.random.rand(1000, 10)
    columns = [f"col_{i}" for i in range(10)]
    return pd.DataFrame(data, columns=columns)

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_load_dataset_positive(tmp_path, valid_dataframe, mock_logger):
    """Positive Test Case: Valid CSV loads correctly."""
    file_path = tmp_path / "valid_data.csv"
    valid_dataframe.to_csv(file_path, index=False)
    
    df = load_dataset(str(file_path))
    
    assert df is not None
    assert not df.empty
    assert df.shape == (1000, 10)

def test_load_dataset_negative_file_not_found(mock_logger):
    """Negative Test Case: File not found."""
    result = load_dataset("non_existent_file.csv")
    assert result is None
    mock_logger.error.assert_called()

def test_load_dataset_boundary_limits(tmp_path, mock_logger):
    """Boundary Test Case: Exactly 1000 rows, 10 columns."""
    df_boundary = pd.DataFrame(np.random.rand(1000, 10), columns=[f"c{i}" for i in range(10)])
    file_path = tmp_path / "boundary_data.csv"
    df_boundary.to_csv(file_path, index=False)

    result = load_dataset(str(file_path))
    assert result is not None
    assert result.shape == (1000, 10)

def test_load_dataset_edge_insufficient_rows(tmp_path, mock_logger):
    """Edge Test Case: 999 rows (should fail)."""
    df_edge = pd.DataFrame(np.random.rand(999, 10), columns=[f"c{i}" for i in range(10)])
    file_path = tmp_path / "edge_rows.csv"
    df_edge.to_csv(file_path, index=False)

    result = load_dataset(str(file_path))
    assert result is None
    # Flexible assertion for any error logged
    mock_logger.error.assert_called()

def test_load_dataset_edge_insufficient_columns(tmp_path, mock_logger):
    """Edge Test Case: 9 columns (should fail)."""
    df_edge = pd.DataFrame(np.random.rand(1000, 9), columns=[f"c{i}" for i in range(9)])
    file_path = tmp_path / "edge_cols.csv"
    df_edge.to_csv(file_path, index=False)

    result = load_dataset(str(file_path))
    assert result is None

def test_load_dataset_null_empty_file(tmp_path, mock_logger):
    """Null Test Case: Empty file."""
    file_path = tmp_path / "empty.csv"
    file_path.touch()

    result = load_dataset(str(file_path))
    assert result is None
    mock_logger.error.assert_called_with("CSV file is empty or contains no data.")

def test_load_dataset_error_corrupted(tmp_path, mock_logger):
    """Error Test Case: Corrupted file triggering ParserError."""
    file_path = tmp_path / "corrupted.csv"
    
    # FIX: We use an unclosed quote. This guarantees a ParserError.
    with open(file_path, "w") as f:
        f.write('col1,col2\n"val1,val2') 

    result = load_dataset(str(file_path))
    
    assert result is None
    mock_logger.error.assert_called_with("CSV file is corrupted or improperly formatted.")