import pandas as pd
from src.logger import setup_logger

# Initialize logger for this module
logger = setup_logger("Outlier_Detection")

def remove_outliers_iqr(df: pd.DataFrame, col_name: str, threshold: float = 1.5) -> pd.DataFrame:
    """
    Detects and removes outliers using the Interquartile Range (IQR) method.
    Returns a cleaned DataFrame without the outliers.
    """
    logger.info(f"Starting outlier detection for column '{col_name}' using IQR.")
    
    try:
        # 1. Validation Checks (Defensive Programming)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
            
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found in the DataFrame.")

        # Ensure column is numeric (cannot calculate IQR on strings). This will raise an AssertionError if False.
        assert pd.api.types.is_numeric_dtype(df[col_name]), \
            f"Column '{col_name}' is not numeric. IQR requires numeric data."

        # 2. Calculation of IQR
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)

        # 3. Identification & Filtering
        mask_valid = (df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)
        
        # Calculate how many rows are being removed
        outliers_count = len(df) - mask_valid.sum()
        
        # Log the exact number of outliers found (This is the line you asked for)
        logger.info(f"Outlier Stats: Found {outliers_count} outliers in '{col_name}' "
                    f"({outliers_count / len(df):.2%} of data).")

        # Safety Check: Warn if we are removing too much data (>10%)
        if outliers_count > len(df) * 0.1:
            logger.warning(f"High outlier count! Removing {outliers_count} rows. "
                           f"Check if threshold ({threshold}) is too strict.")

        # 4. Create Clean DataFrame for valid values (within bounds)
        df_clean = df[mask_valid].copy()

        logger.info(f"SUCCESS: Removed {outliers_count} outliers from '{col_name}'.")
        return df_clean

    # Specific Error Handling (Secure & Robust)
    except TypeError as e:
        logger.error(f"Usage Error (Type Mismatch): {e}")
        return df

    except KeyError as e:
        logger.error(f"Input Error (Missing Column): {e}")
        return df

    except AssertionError as e:
        logger.error(f"Integrity Check Failed (Data Logic): {e}")
        return df
    
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.exception(f"Critical error during outlier removal in '{col_name}': {e}")
        return df