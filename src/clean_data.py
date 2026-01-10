import pandas as pd
from src.logger import setup_logger  

logger = setup_logger("Data_Cleaning") # Create a unique logger for this file/module

def remove_irrelevant_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame if they exist.
    Useful for cleaning non-predictive features (e.g., IDs) before analysis.
    """
    logger.info(f"START: Attempting to remove columns: {columns_to_remove}")    
    
    # --- Input Validation ---
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    assert isinstance(columns_to_remove, list), "Input 'columns_to_remove' must be a list (e.g., ['id'])."
    
    # Create a copy to avoid SettingWithCopyWarning on original dataframe
    df_clean = df.copy()

    if not columns_to_remove:
            logger.warning("No columns provided to remove (empty list).")
            return df_clean
    try:
        # --- Identify which columns actually exist ---
        existing_cols = []
        missing_cols = []  
        
        # Iterate over the requested columns and check if they exist in the dataset
        for col in columns_to_remove:
            if col in df_clean.columns:
                existing_cols.append(col)
            else:
                missing_cols.append(col)
                
        if missing_cols:
            logger.warning(f"Note: The following columns were not found and could not be dropped: {missing_cols}")
        
        # Check for case mismatches and suggest corrections    
        current_columns_lower = {c.lower(): c for c in df_clean.columns} 
        for missing in missing_cols:
                if missing.lower() in current_columns_lower:
                    suggestion = current_columns_lower[missing.lower()]
                    logger.info(f"Column '{missing}' not found. Mabey you mean '{suggestion}'")
                    
        if existing_cols:
            # Perform the drop operation
            df_clean = df_clean.drop(columns=existing_cols)
            logger.info(f"Successfully dropped columns: {existing_cols}")

        for col in columns_to_remove:
            # Verify that the columns were actually removed
            assert col not in df_clean.columns, f"Critical Error: Column '{col}' is still present after dropping!"
        
        return df_clean

    except Exception as e:
        # Log the specific error and re-raise it to stop execution if critical
        logger.error(f"Critical error in drop_irrelevant_columns: {e}")
        raise e
    

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
        # 1. Pre-check: Validate that the column exists
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' does not exist in the DataFrame.")

        # 2. Pre-check: Validate Numeric Type
        assert pd.api.types.is_numeric_dtype(df[col_name]), \
            f"Prerequisite Failed: Column '{col_name}' is of type {df[col_name].dtype}, but must be numeric."

        # 3. Calculation Logic
        median_val = df[col_name].median()

        assert not pd.isna(median_val), \
            f"Calculation Error: Median for '{col_name}' is NaN (Column might be entirely empty)."

        # 4. Apply Changes
        initial_nans = df[col_name].isna().sum()
        
        if initial_nans > 0:
            df[col_name] = df[col_name].fillna(median_val)
            logger.info(f"Action: Filled {initial_nans} missing values with median ({median_val:.2f}).")
        else:
            logger.info(f"No Action: Column '{col_name}' has no missing values.")

        # 5. Final Verification (Post-Check)
        remaining_nans = df[col_name].isna().sum()
        assert remaining_nans == 0, \
            f"Verification Failed: Column '{col_name}' still has {remaining_nans} missing values."

        logger.info(f"SUCCESS: Column '{col_name}' processing complete.")
        return df

    # --- Exception Handling ---

    except KeyError as e:
        logger.error(f"Input Error: {e}")
        return df

    except AssertionError as e:
        # תופס את כל ה-Asserts שכתבנו (לא מספרי, חציון לא תקין, וכו')
        logger.error(f"Integrity Check Failed: {e}")
        return df

    except Exception as e:
        # תופס שגיאות לא צפויות אחרות
        logger.exception(f"Unexpected error while filling '{col_name}': {e}")
        return df    
    
    
    
    
    
    
    
    