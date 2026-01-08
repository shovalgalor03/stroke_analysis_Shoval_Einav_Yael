import pandas as pd
import os
from src.logger import setup_logger  # <--- Change: Import the central logger

# <--- Change: Create a unique logger for this file
logger = setup_logger("Data_Loading")

def load_dataset(file_path):
    """
    Safely loads a CSV file into a pandas DataFrame and verifies basic data validity.

    The function checks that the file exists, attempts to read it, and ensures
    that the loaded dataset is not empty and contains a valid structure.

    Parameters
    ----------
    file_path : str
        Full path to the CSV file to be loaded.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the dataset if loading is successful;
        otherwise, None is returned.
    """
    try:
        # Check that the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # <--- Change: Log process start
        logger.info(f"Attempting to load dataset from {file_path}")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Integrity checks
        assert not df.empty, "Dataset is empty."
        assert df.shape[1] >= 10, "Dataset must contain at least 10 columns to meet the project requirements."
        
        # Added check for row count based on project checklist
        assert df.shape[0] >= 1000, "Dataset must contain at least 1000 rows to meet the project requirements."

        # <--- Change: Log success with shape info
        logger.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )

        return df

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")

    except AssertionError as e:
        logger.error(f"Data integrity check failed: {e}")

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or contains no data.")

    except pd.errors.ParserError:
        logger.error("CSV file is corrupted or improperly formatted.")

    except Exception as e:
        # exception() logs the full traceback
        logger.exception(f"Unexpected error while loading dataset: {e}")

    return None