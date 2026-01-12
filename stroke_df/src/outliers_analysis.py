import pandas as pd
from src.logger import setup_logger

logger = setup_logger("Outlier_Detection")

# Function 1: Calculation & Validation

def calculate_iqr_bounds(df: pd.DataFrame, col_name: str, threshold: float) -> tuple:
    """
    Validates input and calculates IQR bounds.
    Returns a tuple (lower_bound, upper_bound) or None if validation fails.
    """
    try:
        # 1. Validation Checks
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found.")
        
        # Ensure column is numeric
        assert pd.api.types.is_numeric_dtype(df[col_name]), \
            f"Column '{col_name}' is not numeric. Cannot calculate IQR."

        # 2. Math Logic
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - (threshold * IQR)
        upper = Q3 + (threshold * IQR)
        
        logger.info(f"IQR Bounds calculated for '{col_name}': [{lower:.2f}, {upper:.2f}]")
        return lower, upper

    # Specific Error Handling for Calculation Phase
    except TypeError as e:
        logger.error(f"Validation Error - Type: {e}")
        return None
    except KeyError as e:
        logger.error(f"Validation Error - Column Missing: {e}")
        return None
    except AssertionError as e:
        logger.error(f"Validation Error - Data Logic: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in IQR calculation: {e}")
        return None
    
# Function 2: Execution (Filtering)

def remove_outliers_iqr(df: pd.DataFrame, col_name: str, threshold: float = 1.5) -> pd.DataFrame:
    """
    Manages the outlier removal process. 
    Calls the calculation function and applies the filter.
    Returns the cleaned dataframe (or the original one if an error occurs).
    """
    logger.info(f"Starting outlier removal for '{col_name}' (Threshold: {threshold})")

    try:
        # Step 1: Get bounds from helper function
        bounds = calculate_iqr_bounds(df, col_name, threshold)
        
        # If calculation failed (returned None), abort safely
        if bounds is None:
            logger.error("Aborting outlier removal due to validation failure.")
            return df

        lower_bound, upper_bound = bounds

        # Step 2: Apply Filter
        # We re-verify column existence implicitly here by accessing it
        mask_valid = (df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)
        outliers_count = len(df) - mask_valid.sum()
        
        # Step 3: Logging & Safety Warning
        logger.info(f"Stats: Removing {outliers_count} outliers ({outliers_count/len(df):.2%} of data).")

        if outliers_count > len(df) * 0.1:
            logger.warning(f"High outlier count! Removing >10% of data from '{col_name}'. Check threshold.")

        # Create a df for valid values (within bounds)
        return df[mask_valid].copy()

    # --- Specific Error Handling for Execution Phase ---
    
    except KeyError as e:
        # Could happen if column is dropped during execution (unlikely but possible)
        logger.error(f"Execution Error - Column lost: {e}")
        return df

    except TypeError as e:
        # Could happen if types mismatch during comparison
        logger.error(f"Execution Error - Type mismatch during filtering: {e}")
        return df

    except Exception as e:
        # Catch-all for any other critical failures
        logger.exception(f"Critical error applying outlier filter: {e}")
        return df