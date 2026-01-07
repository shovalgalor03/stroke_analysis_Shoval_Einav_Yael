import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    filename="data_loading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

        logging.info(f"Attempting to load dataset from {file_path}")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Integrity checks
        assert not df.empty, "Dataset is empty."
        assert df.shape[1] >= 10, "Dataset must contain at least 10 columns to meet the project requirements."

        logging.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )

        return df

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")

    except AssertionError as e:
        logging.error(f"Data integrity check failed: {e}")

    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty or contains no data.")

    except pd.errors.ParserError:
        logging.error("CSV file is corrupted or improperly formatted.")

    except Exception as e:
        logging.exception(f"Unexpected error while loading dataset: {e}")

    return None