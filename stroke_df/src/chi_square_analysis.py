import pandas as pd
from scipy.stats import chi2_contingency
from src.logger import setup_logger

# Initialize the logger
logger = setup_logger("Chi_Square_Analysis")

# Function 1: Data Preparation

def create_contingency_table(df: pd.DataFrame, var1: str, var2: str) -> pd.DataFrame:
    """
    Validates input data, handles missing values, and generates a contingency table.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        var1 (str): The name of the first categorical column.
        var2 (str): The name of the second categorical column.

    Returns:
        pd.DataFrame: A contingency table (crosstab) if successful, or None if validation fails.
    """
    try:
        # 1. Basic Validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
            
        if var1 not in df.columns or var2 not in df.columns:
            raise KeyError(f"Columns '{var1}' or '{var2}' not found in DataFrame.")

        if df.empty:
            raise ValueError("DataFrame is empty.")

        # 2. Handle Missing Values (Data Cleaning)
        # Create a copy to avoid modifying the original dataframe
        df_clean = df[[var1, var2]].dropna()
        
        dropped_rows = len(df) - len(df_clean)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to NaN values in '{var1}' or '{var2}'.")

        assert len(df_clean) > 0, "No valid data remaining after cleaning NaNs."

        # 3. Create the Contingency Table (Crosstab)
        contingency_table = pd.crosstab(df_clean[var1], df_clean[var2])
        
        # Verify table size (must be at least 2x2 for Chi-Square)
        assert contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2, \
            "Contingency table is too small (needs at least 2x2 categories)."

        logger.info(f"Contingency table created successfully. Shape: {contingency_table.shape}")
        return contingency_table

    # --- Specific Error Handling for Data Prep ---
    except KeyError as e:
        logger.error(f"Data Prep Error - Column not found: {e}")
        return None

    except TypeError as e:
        logger.error(f"Data Prep Error - Invalid input type: {e}")
        return None

    except AssertionError as e:
        logger.error(f"Data Prep Error - Data quality issue: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error creating contingency table: {e}")
        return None

# Function 2: Statistical Execution

def run_chi_square_test(df: pd.DataFrame, independent_var: str, dependent_var: str, alpha: float = 0.05) -> dict:
    """
    Orchestrates the Chi-Square Test of Independence.
    It calls the preparation function first, then performs the statistical test.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        independent_var (str): The grouping variable.
        dependent_var (str): The target variable.
        alpha (float): Significance level (default 0.05).

    Returns:
        dict: A dictionary with test results and interpretation, or None if failed.
    """
    logger.info(f"--- Starting Chi-Square Test: {independent_var} vs {dependent_var} ---")

    try:
        # Step 1: Get the prepared contingency table using the helper function
        contingency_table = create_contingency_table(df, independent_var, dependent_var)

        # If data preparation failed (returned None), we stop here safely
        if contingency_table is None:
            logger.error("Aborting test due to data preparation failure.")
            return None

        # Step 2: Statistical Calculation
        chi2, p_val, dof, expected_freq = chi2_contingency(contingency_table)

        # Step 3: Interpretation & Assumption Checks
        
        # Check assumption: Expected frequencies should be > 5
        if (expected_freq < 5).any():
            logger.warning("Assumption Warning: Some expected frequencies are < 5. Results may be less accurate.")

        # Determine significance
        # This explicitly checks if alpha is valid during comparison
        is_significant = p_val < alpha
        result_text = "Significant (Reject H0)" if is_significant else "Not Significant (Fail to Reject H0)"

        logger.info(f"Test Finished. P-value: {p_val:.5f} | Result: {result_text}")

        return {
            "chi2_statistic": chi2,
            "p_value": p_val,
            "degrees_of_freedom": dof,
            "significant": is_significant,
            "interpretation": result_text,
            "contingency_table": contingency_table 
        }

    # --- Specific Error Handling for Execution Phase ---
    
    except TypeError as e:
        # Catches errors like comparing p_val (float) with alpha (string)
        logger.error(f"Execution Error - Type Mismatch: {e}")
        return None

    except ValueError as e:
        # Catches mathematical errors from scipy (e.g., if table contains invalid values)
        logger.error(f"Execution Error - Value Error: {e}")
        return None

    except Exception as e:
        # Catches any other unexpected errors
        logger.exception(f"Critical error during statistical calculation: {e}")
        return None