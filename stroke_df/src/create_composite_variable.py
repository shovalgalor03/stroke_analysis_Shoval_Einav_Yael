import pandas as pd
import numpy as np
from src.logger import setup_logger

logger = setup_logger("composite_module") #Create a unique logger for this file
# --- Initialize Logger (immediately after imports) ---
logger = setup_logger("composite_module")

def create_composite_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a composite 'risk_group' variable based on binary flags.
    Based on the roadmap: combining BMI and Glucose flags into 4 distinct groups.
    Includes defensive programming checks.
    """
    # --- 1. Pre-validation (Assertions) ---
    # Validate input type
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

    # [cite_start]Validate that the necessary flags from the previous step exist [cite: 45, 262-263]
    required_flags = ['bmi_high', 'glucose_high']
    for col in required_flags:
        assert col in df.columns, f"Missing required column: '{col}'. Run 'convert_continuous_to_categorical' first."

    try:
        # Work on a copy
        df_comp = df.copy()

        # --- 2. Data Integrity Check ---
        # Ensure flags contain only 0 and 1 (validating the previous step worked correctly)
        for col in required_flags:
            unique_vals = df_comp[col].unique()
            # Assert that all values are either 0 or 1
            assert set(unique_vals).issubset({0, 1}), f"Column '{col}' contains invalid values (must be 0 or 1)."

        # --- 3. Logic Implementation ---
        # [cite_start]Define conditions for the 4 groups [cite: 46, 264-265]
        # 1. Both High
        cond_both = (df_comp['bmi_high'] == 1) & (df_comp['glucose_high'] == 1)
        # 2. BMI Only (High BMI but Normal Glucose)
        cond_bmi_only = (df_comp['bmi_high'] == 1) & (df_comp['glucose_high'] == 0)
        # 3. Glucose Only (Normal BMI but High Glucose)
        cond_glucose_only = (df_comp['bmi_high'] == 0) & (df_comp['glucose_high'] == 1)
        # 4. Neither (Control group)
        cond_neither = (df_comp['bmi_high'] == 0) & (df_comp['glucose_high'] == 0)

        # Apply logic using np.select (Faster and safer than .apply)
        conditions = [cond_both, cond_bmi_only, cond_glucose_only, cond_neither]
        choices = ['both_high', 'bmi_only', 'glucose_only', 'neither']
        
        df_comp['risk_group'] = np.select(conditions, choices, default='ERROR')

        # --- 4. Post-validation (Sanity Check) ---
        # Verify no rows were missed (no 'ERROR' values)
        if 'ERROR' in df_comp['risk_group'].values:
            error_count = (df_comp['risk_group'] == 'ERROR').sum()
            raise ValueError(f"Logic Error: {error_count} rows could not be assigned to a group.")
        
        # Verify the new column exists
        assert 'risk_group' in df_comp.columns, "Failed to create 'risk_group' column."
        
        # [cite_start][Optional] Log the distribution to ensure groups are balanced/exist [cite: 266]
        logger.info("Composite Variable Created Successfully. Group Distribution:\n" + str(df_comp['risk_group'].value_counts()))

        return df_comp

    except Exception as e:
        # Changed print to logger.error for consistency
        logger.error(f"CRITICAL ERROR in create_composite_variable: {str(e)}")
        raise e