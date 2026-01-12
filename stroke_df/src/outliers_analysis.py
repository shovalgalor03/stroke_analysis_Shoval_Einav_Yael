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

        # Ensure column is numeric (cannot calculate IQR on strings)
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
        # Create a mask for valid values (within bounds)
        mask_valid = (df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)
        
        # Calculate how many rows are being removed
        outliers_count = len(df) - mask_valid.sum()
        
        # Safety Check: Warn if we are removing too much data (>10%)
        if outliers_count > len(df) * 0.1:
            logger.warning(f"High outlier count! Removing {outliers_count} rows ({outliers_count/len(df):.1%}).")

        # 4. Create Clean DataFrame
        df_clean = df[mask_valid].copy()

        logger.info(f"Outlier removal complete. Removed {outliers_count} outliers from '{col_name}'.")
        return df_clean

    except Exception as e:
        # Log the full error and return the original dataframe to avoid crashing the pipeline
        logger.exception(f"Error during outlier removal in '{col_name}': {e}")
        return df