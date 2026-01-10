import pandas as pd
from src.logger import setup_logger  

logger = setup_logger("remove_columns") # Create a unique logger for this file/module

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