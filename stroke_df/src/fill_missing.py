import pandas as pd
from src.logger import setup_logger  

logger = setup_logger("fill_missing") # Create a unique logger for this file/module

def fill_missing_with_median(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Fills missing values (NaN) in a numeric column using the median value.
    Assumes the column has already been converted to numeric.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to fill.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled.
                      Returns original DataFrame if validation fails.
    """
    
    logger.info(f"START: Filling missing values for '{col_name}' using Median.")

    try:
        df_filled = df.copy()  # Create a copy to avoid SettingWithCopyWarning on original dataframe
        
        # 1. Pre-check: Validate that the column exists
        if col_name not in df_filled.columns:
            raise KeyError(f"Column '{col_name}' does not exist in the DataFrame.")

        # 2. Pre-check: Validate Numeric Type
        assert pd.api.types.is_numeric_dtype(df_filled[col_name]), \
            f"Prerequisite Failed: Column '{col_name}' is of type {df_filled[col_name].dtype}, but must be numeric."

        # 3. Calculation Logic
        median_val = df_filled[col_name].median()

        assert not pd.isna(median_val), \
            f"Calculation Error: Median for '{col_name}' is NaN (Column might be entirely empty)."

        # 4. Apply Changes
        initial_nans = df_filled[col_name].isna().sum()
        
        if initial_nans > 0:
            df_filled[col_name] = df_filled[col_name].fillna(median_val)
            logger.info(f"Action: Filled {initial_nans} missing values with median ({median_val:.2f}).")
        else:
            logger.info(f"No Action: Column '{col_name}' has no missing values.")

        # 5. Final Verification (Post-Check)
        remaining_nans = df_filled[col_name].isna().sum()
        assert remaining_nans == 0, \
            f"Verification Failed: Column '{col_name}' still has {remaining_nans} missing values."

        logger.info(f"SUCCESS: Column '{col_name}' processing complete.")
        return df_filled

    # --- Exception Handling ---
    except KeyError as e:
        logger.error(f"Input Error: {e}")
        return df

    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        return df

    except Exception as e:
        logger.exception(f"Unexpected error while filling '{col_name}': {e}")
        return df    
    
    