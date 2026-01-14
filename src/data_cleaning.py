import pandas as pd
from src.logger import setup_logger  # Import the central logger

logger = setup_logger("Data_Cleaning") # Create a unique logger for this module

def remove_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame if they exist.
    Useful for cleaning non-predictive features (e.g., IDs) before analysis.
    """
    logger.info(f"START: Attempting to remove columns: {columns_to_remove}")    
    
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        
        if not isinstance(columns_to_remove, list):
            raise TypeError(f"Input 'columns_to_remove' must be a list, got {type(columns_to_remove)}.")
        
        df_clean = df.copy() # Create a copy to avoid SettingWithCopyWarning on original dataframe

        if not columns_to_remove:
                logger.warning("No columns provided to remove (empty list).")
                return df_clean
            
        existing_cols = []
        missing_cols = []  
        
        for col in columns_to_remove:  # Iterate over the requested columns and check if they exist in the dataset
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
                    logger.info(f"Column '{missing}' not found. Maybe you mean '{suggestion}'")
                    
        if existing_cols: # Perform the drop operation
            df_clean = df_clean.drop(columns=existing_cols)
            logger.info(f"Successfully dropped columns: {existing_cols}")

        for col in columns_to_remove: # Verify that the columns were actually removed
            assert col not in df_clean.columns, f"Critical Error: Column '{col}' is still present after dropping!"
        
        return df_clean

    except TypeError as e:
        logger.error(f"Usage Error: {e}")
        return df

    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        return df
    
    except KeyError as e:
        logger.error(f"Input Error: {e}")
        return df
    
    except Exception as e:
        logger.error(f"Critical error in drop_irrelevant_columns: {e}")
        return df

def convert_to_numeric(df: pd.DataFrame, col_name: str, safety_threshold: float = 0.5) -> pd.DataFrame:
    """
    Safely converts a DataFrame column to numeric type (float).
    Non-numeric values (e.g., 'N/A', 'Error', strings) are coerced to NaN.

    This function works on a COPY of the DataFrame to prevent side effects.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to convert.
        safety_threshold (float): The maximum allowed ratio of data becoming NaN 
                                  during conversion (0.0 to 1.0). Default is 0.5 (50%).

    Returns:
        pd.DataFrame: A new DataFrame with the converted column.
                      Returns original DataFrame (copy) if an error occurs.
    """
    
    logger.info(f"START: processing column '{col_name}'.")

    df_clean = df.copy() # Create a copy to ensure we don't modify the original dataframe in case of error mid-process
    
    try:
        # 1. Pre-check: Validate that the column exists
        if col_name not in df_clean.columns:
            raise KeyError(f"Column '{col_name}' does not exist in the DataFrame.")

        # Capture initial state for logging comparison
        initial_nans = df_clean[col_name].isna().sum()
        total_rows = len(df_clean)

        # 2. Conversion Logic
        converted_series = pd.to_numeric(df_clean[col_name], errors='coerce') # 'coerce' forces invalid strings to become NaN

        # 3. Safety Guard (Assertion)
        current_nans = converted_series.isna().sum()
        newly_created_nans = current_nans - initial_nans
        loss_ratio = newly_created_nans / total_rows if total_rows > 0 else 0
        # Stop if too much data became NaN
        assert loss_ratio < safety_threshold, \
            f"Safety Stop: Conversion resulted in {loss_ratio:.1%} data loss. Threshold is {safety_threshold:.1%}."

        # 4. Apply Changes
        df_clean[col_name] = converted_series

        # 5. Final Type Check
        assert pd.api.types.is_numeric_dtype(df_clean[col_name]), \
            f"Verification Failed: Column '{col_name}' is still not numeric."

        logger.info(f"SUCCESS: Column '{col_name}' converted. {newly_created_nans} invalid values set to NaN.")
        return df_clean

    except KeyError as e:
        logger.error(f"Input Error: {e}")
        return df 

    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        return df 

    except Exception as e:
        logger.exception(f"Unexpected error while converting '{col_name}': {e}")
        return df
    
def fill_missing_with_median(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Fills missing values (NaN) in a numeric column using the median value.
    Assumes the column has already been converted to numeric.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to fill.

    Returns:
        pd.DataFrame: A Copy of the DataFrame with missing values filled.
                      Returns original DataFrame if validation fails.
    """
    
    logger.info(f"START: Filling missing values for '{col_name}' using Median.")

    df_filled = df.copy() # Create a copy to work safely

    try:        
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

    except KeyError as e:
        logger.error(f"Input Error: {e}")
        return df

    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        return df

    except Exception as e:
        logger.exception(f"Unexpected error while filling '{col_name}': {e}")
        return df    
        