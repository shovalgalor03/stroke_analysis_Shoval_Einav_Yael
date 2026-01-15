import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from src.logger import setup_logger
from src.constants import BMI_THRESHOLD, GLUCOSE_THRESHOLD
logger = setup_logger("Feature_Engineering") # Initialize a single logger for this module

def convert_continuous_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts continuous variables (bmi, avg_glucose_level) into categorical/binary flags.
    Includes safety checks (assertions and try-except).
    """
    # --- 1. Pre-validation (Assertions) ---
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    
    required_cols = ['bmi', 'avg_glucose_level']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: '{col}' in DataFrame."

    try:
        logger.info("Starting conversion of continuous variables to binary flags...")

        # Create a copy to avoid SettingWithCopyWarning
        df_cat = df.copy()

        # --- 2. Data Type Validation ---
        for col in required_cols:
             if not is_numeric_dtype(df_cat[col]):
                 raise TypeError(f"Column '{col}' must be numeric type. Please clean data first.")

        # Create binary flags using np.where (Safe & Fast)
        df_cat['bmi_high'] = np.where(df_cat['bmi'] >= BMI_THRESHOLD, 1, 0)
        df_cat['glucose_high'] = np.where(df_cat['avg_glucose_level'] >= GLUCOSE_THRESHOLD, 1, 0)

        # Define internal function for grouping
        def get_group(row):
            b_high = row.get('bmi_high', 0)
            g_high = row.get('glucose_high', 0)
            
            if b_high == 1 and g_high == 1:
                return 'both_high'
            elif b_high == 1:
                return 'bmi_only'
            elif g_high == 1:
                return 'glucose_only'
            else:
                return 'neither'

        # Apply grouping 
        df_cat['risk_group'] = df_cat.apply(get_group, axis=1)

        # --- 4. Post-validation (Output Check) ---
        assert 'risk_group' in df_cat.columns, "Failed to create 'risk_group' column."
        
        if df_cat['risk_group'].isnull().any():
             logger.warning("Null values found in 'risk_group' column after conversion.")

        logger.info("Successfully created 'risk_group' column.")
        return df_cat

    except Exception as e:
        logger.error(f"CRITICAL ERROR in convert_continuous_to_categorical: {str(e)}")
        raise e


def create_composite_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a composite 'risk_group' variable based on binary flags.
    Based on the roadmap: combining BMI and Glucose flags into 4 distinct groups.
    """
    # --- 1. Pre-validation ---
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

    required_flags = ['bmi_high', 'glucose_high']
    for col in required_flags:
        assert col in df.columns, f"Missing required column: '{col}'. Run 'convert_continuous_to_categorical' first."

    try:
        df_comp = df.copy() 

        # --- 2. Data Integrity Check ---
        for col in required_flags:
            unique_vals = df_comp[col].unique()
            assert set(unique_vals).issubset({0, 1}), f"Column '{col}' contains invalid values (must be 0 or 1)."

        # --- 3. Logic Implementation ---
        # 1. Both High
        cond_both = (df_comp['bmi_high'] == 1) & (df_comp['glucose_high'] == 1)
        # 2. BMI Only
        cond_bmi_only = (df_comp['bmi_high'] == 1) & (df_comp['glucose_high'] == 0)
        # 3. Glucose Only
        cond_glucose_only = (df_comp['bmi_high'] == 0) & (df_comp['glucose_high'] == 1)
        # 4. Neither
        cond_neither = (df_comp['bmi_high'] == 0) & (df_comp['glucose_high'] == 0)

        # Apply logic using np.select
        conditions = [cond_both, cond_bmi_only, cond_glucose_only, cond_neither]
        choices = ['both_high', 'bmi_only', 'glucose_only', 'neither']
        
        df_comp['risk_group'] = np.select(conditions, choices, default='ERROR')

        # --- 4. Post-validation ---
        if 'ERROR' in df_comp['risk_group'].values:
            error_count = (df_comp['risk_group'] == 'ERROR').sum()
            raise ValueError(f"Logic Error: {error_count} rows could not be assigned to a group.")
        
        assert 'risk_group' in df_comp.columns, "Failed to create 'risk_group' column."
        
        # Log distribution instead of print
        dist = df_comp['risk_group'].value_counts().to_dict()
        logger.info(f"Composite Variable Created. Group Distribution: {dist}")

        return df_comp

    except Exception as e:
        logger.error(f"CRITICAL ERROR in create_composite_variable: {str(e)}")
        raise e