import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from src.logger import setup_logger
from src.constants import BMI_THRESHOLD, GLUCOSE_THRESHOLD

logger = setup_logger("Feature_Engineering") 

def convert_continuous_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts continuous variables (bmi, avg_glucose_level) into categorical/binary.
    """
    #  Assertion - check if the input is in df shape.
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    
    # Checks if the required columns are in df.
    required_cols = ['bmi', 'avg_glucose_level'] 
    for col in required_cols:
        assert col in df.columns, f"Missing required column: '{col}' in DataFrame."

    try:
        logger.info("Starting conversion of continuous variables to binary flags.")

        # Creates a copy to avoid tempering with the original df.
        df_cat = df.copy()

        #  Data type validation - checks if the data in the required columns is numeric.
        for col in required_cols:
             if not is_numeric_dtype(df_cat[col]):
                 raise TypeError(f"Column '{col}' must be numeric type. Please clean data first.")

        # Creates binary variable - assigns 1 if higher than threshold, else 0.
        df_cat['bmi_high'] = np.where(df_cat['bmi'] >= BMI_THRESHOLD, 1, 0)
        df_cat['glucose_high'] = np.where(df_cat['avg_glucose_level'] >= GLUCOSE_THRESHOLD, 1, 0)

        # Define internal function for grouping. If the column is missing/empty, the value is 0 to prevent error.
        def get_group(row):
            b_high = row.get('bmi_high', 0)
            g_high = row.get('glucose_high', 0)

            # Creates the composite variable - risk groups.
            if b_high == 1 and g_high == 1:
                return 'both_high'
            elif b_high == 1:
                return 'bmi_only'
            elif g_high == 1:
                return 'glucose_only'
            else:
                return 'neither'

        # Apply grouping. 
        df_cat['risk_group'] = df_cat.apply(get_group, axis=1)

        # Assertion to check if was able to create the composite variable.
        assert 'risk_group' in df_cat.columns, "Failed to create 'risk_group' column."
        
        # Checks for any missing values (nulls) in 'risk_group' column.
        if df_cat['risk_group'].isnull().any():
             logger.warning("Null values found in 'risk_group' column after conversion.")

        logger.info("Successfully created 'risk_group' column.")
        return df_cat

    except Exception as e:
        logger.error(f"CRITICAL ERROR in convert_continuous_to_categorical: {str(e)}")
        raise e


def create_composite_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a composite 'risk_group' variable based on binarization.
    Combining BMI and Glucose into 4 risk groups.
    """
    # Assertion - check if the input is in df shape.
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

    # Checks if the required columns are in df.
    required_flags = ['bmi_high', 'glucose_high']
    for col in required_flags:
        assert col in df.columns, f"Missing required column: '{col}'. Run 'convert_continuous_to_categorical' first."

    try:
        df_comp = df.copy() # creates a copy of df.

        # Data integrity check - verifies that all columns are binary (0 or 1).
        for col in required_flags:
            unique_vals = df_comp[col].unique()
            assert set(unique_vals).issubset({0, 1}), f"Column '{col}' contains invalid values (must be 0 or 1)."

        # Logic implementation - creates the 4 conditions.
        # 1. Both High
        cond_both = (df_comp['bmi_high'] == 1) & (df_comp['glucose_high'] == 1)
        # 2. BMI Only
        cond_bmi_only = (df_comp['bmi_high'] == 1) & (df_comp['glucose_high'] == 0)
        # 3. Glucose Only
        cond_glucose_only = (df_comp['bmi_high'] == 0) & (df_comp['glucose_high'] == 1)
        # 4. Neither
        cond_neither = (df_comp['bmi_high'] == 0) & (df_comp['glucose_high'] == 0)

        # Applies logic using np.select - matches specific conditions to their corresponding labels. If no condition matches, assigns 'ERROR'.
        conditions = [cond_both, cond_bmi_only, cond_glucose_only, cond_neither]
        choices = ['both_high', 'bmi_only', 'glucose_only', 'neither']
        df_comp['risk_group'] = np.select(conditions, choices, default='ERROR')

        # Counts the num of 'ERROR' to identify how many rows don't have group.
        if 'ERROR' in df_comp['risk_group'].values:
            error_count = (df_comp['risk_group'] == 'ERROR').sum()
            raise ValueError(f"Logic Error: {error_count} rows could not be assigned to a group.")
        
        assert 'risk_group' in df_comp.columns, "Failed to create 'risk_group' column."
        
        # Calculates the frequency of each group.
        dist = df_comp['risk_group'].value_counts().to_dict()
        logger.info(f"Composite Variable Created. Group Distribution: {dist}")

        return df_comp

    except Exception as e:
        logger.error(f"CRITICAL ERROR in create_composite_variable: {str(e)}")
        raise e