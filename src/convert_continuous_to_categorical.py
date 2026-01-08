import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from src.utils import setup_logger  # <--- השינוי: ייבוא הלוגר

# <--- השינוי: יצירת הלוגר עבור הקובץ הזה
logger = setup_logger("Data_Conversion")

def convert_continuous_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts continuous variables (bmi, avg_glucose_level) into categorical/binary flags
    and creates a combined group variable based on the project roadmap.
    Includes safety checks (assertions and try-except).
    """
    # --- 1. Pre-validation (Assertions) ---
    # Validate input type
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    
    # Validate required columns exist 
    required_cols = ['bmi', 'avg_glucose_level']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: '{col}' in DataFrame."

    try:
        # <--- השינוי: תיעוד התחלת התהליך
        logger.info("Starting conversion of continuous variables to categorical categories...")

        # Create a copy to avoid SettingWithCopyWarning
        df_cat = df.copy()

        # --- 2. Data Type Validation ---
        # Ensure columns are numeric before applying thresholds
        for col in required_cols:
             if not is_numeric_dtype(df_cat[col]):
                 # כאן אפשר להוסיף לוג שגיאה לפני ה-raise אם רוצים, אבל ה-except למטה יתפוס את זה בכל מקרה
                 raise TypeError(f"Column '{col}' must be numeric type. Please clean data first.")

        # --- 3. Logic Implementation ---
        # Constants from Roadmap 
        BMI_THRESHOLD = 30
        GLUCOSE_THRESHOLD = 140

        # Create binary flags using np.where (Safe & Fast)
        df_cat['bmi_high'] = np.where(df_cat['bmi'] >= BMI_THRESHOLD, 1, 0)
        df_cat['glucose_high'] = np.where(df_cat['avg_glucose_level'] >= GLUCOSE_THRESHOLD, 1, 0)

        # Define internal function for grouping
        def get_group(row):
            # Using .get() adds another layer of safety
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
        # Verify the new column was actually created and is not empty
        assert 'risk_group' in df_cat.columns, "Failed to create 'risk_group' column."
        
        # Optional: Check for nulls in the new column
        if df_cat['risk_group'].isnull().any():
             # <--- השינוי: שימוש ב-warning במקום print
             logger.warning("Null values found in 'risk_group' column after conversion.")

        # <--- השינוי: תיעוד הצלחה
        logger.info("Successfully created 'risk_group' column.")

        return df_cat

    except Exception as e:
        # Catch unexpected errors, log as error, and re-raise
        # <--- השינוי: שימוש ב-error במקום print
        logger.error(f"CRITICAL ERROR in convert_continuous_to_categorical: {str(e)}")
        raise e