import pandas as pd
from src.logger import setup_logger  # Import the central logger

logger = setup_logger("Data_Processing_Numeric") # Create a unique logger for this file

def safe_convert_to_numeric(df: pd.DataFrame, col_name: str, safety_threshold: float = 0.5) -> pd.DataFrame:
    """
    Safely converts a DataFrame column to numeric type (float).
    Non-numeric values (e.g., 'N/A', 'Error', strings) are coerced to NaN.

    This function preserves the DataFrame's original shape (no rows are deleted).
    It includes a safety mechanism to prevent accidental data destruction.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to convert.
        safety_threshold (float): The maximum allowed ratio of data becoming NaN 
                                  during conversion (0.0 to 1.0). Default is 0.5 (50%).

    Returns:
        pd.DataFrame: The DataFrame with the specific column converted to numeric.
                      Returns the original DataFrame if an error occurs.
    """
    
    logger.info(f"START: processing column '{col_name}'.")

    try:
        # 1. Pre-check: Validate that the column exists
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' does not exist in the DataFrame.")

        # Capture initial state for logging comparison
        initial_nans = df[col_name].isna().sum()
        total_rows = len(df)

        # 2. Conversion Logic
        # 'coerce' forces invalid strings to become NaN instead of raising an error.
        # We store this in a temporary variable first.
        converted_series = pd.to_numeric(df[col_name], errors='coerce')

        # 3. Safety Guard (Assertion)
        # Calculate how much data was "lost" (converted to NaN)
        current_nans = converted_series.isna().sum()
        newly_created_nans = current_nans - initial_nans
        loss_ratio = newly_created_nans / total_rows if total_rows > 0 else 0

        # Stop if too much data became NaN (indicates wrong column or bad data)
        assert loss_ratio < safety_threshold, \
            f"Safety Stop: Conversion resulted in {loss_ratio:.1%} data loss. Threshold is {safety_threshold:.1%}."

        # 4. Apply Changes
        # Update the DataFrame column (Shape of df remains unchanged)
        df[col_name] = converted_series

        # 5. Final Type Check
        assert pd.api.types.is_numeric_dtype(df[col_name]), \
            f"Verification Failed: Column '{col_name}' is still not numeric."

        logger.info(f"SUCCESS: Column '{col_name}' converted. {newly_created_nans} invalid values were set to NaN.")
        return df

    # --- Exception Handling ---

    except KeyError as e:
        logger.error(f"Input Error: {e}")

    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")

    except Exception as e:
        # exception to capture the full traceback
        logger.exception(f"Unexpected error while converting '{col_name}': {e}")

        # Return original DataFrame in case of failure to maintain continuity
        return df