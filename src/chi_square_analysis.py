import pandas as pd
from scipy.stats import chi2_contingency
from src.logger import setup_logger  #Importing central logger

#Create a unique logger for this specific analysis module
logger = setup_logger("Chi_Square_Analysis")

def run_chi_square_test(df: pd.DataFrame, independent_var: str, dependent_var: str, alpha: float = 0.05) -> dict:
    """
    Performs a Chi-Square Test of Independence between two categorical variables.
    
    This function verifies assumptions, creates a contingency table, and executes
    the statistical test. It includes safety checks for data integrity and 
    logs all steps for reproducibility.

    Parameters:
    df : pd.DataFrame
        The dataframe containing the data.
    independent_var : str
        The name of the grouping column.
    dependent_var : str
        The name of the target column.
    alpha : float, optional
        The significance level threshold (default is 0.05).

    Returns:
    dict or None
        A dictionary containing the test results (Chi2 stat, p-value, formatting)
        if successful; otherwise, None.
    """
    try:
        logger.info(f"Starting Chi-Square test between '{independent_var}' and '{dependent_var}'.")

        # 1. Pre-validation (Assertions & Checks) 
        
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        # Validate that columns exist in the DataFrame
        if independent_var not in df.columns or dependent_var not in df.columns:
            raise KeyError(f"Columns '{independent_var}' or '{dependent_var}' not found in DataFrame.")

        # Ensure the DataFrame is not empty
        assert not df.empty, "DataFrame is empty. Cannot perform analysis."

        # Check for null values in the relevant columns
        # Statistical tests fail or give wrong results with NaNs.
        null_count = df[[independent_var, dependent_var]].isna().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} missing values. Rows with NaNs will be excluded from the test.")
            # Create a clean subset for the test (does not affect original df)
            df_clean = df.dropna(subset=[independent_var, dependent_var])
        else:
            df_clean = df

        # Safety Check: Ensure we have data left after cleaning
        assert len(df_clean) > 0, "No valid data remaining after removing missing values."

        # 2. Data Preparation 

        # Create the Contingency Table (Observed Frequencies)
        contingency_table = pd.crosstab(df_clean[independent_var], df_clean[dependent_var])
        
        #Log the shape of the table
        logger.info(f"Contingency table created. Shape: {contingency_table.shape}")

        #Assert that the table is at least 2x2 (required for Chi-Square)
        assert contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2, \
            "Contingency table is too small (needs at least 2x2). Check your categories."

        # 3. Statistical Calculation

        # Run the test using Scipy
        chi2, p_val, dof, expected_freq = chi2_contingency(contingency_table)

        # 4. Interpretation & Validation 

        #Check statistical assumption: Expected frequencies should be > 5
        if (expected_freq < 5).any():
            logger.warning("Assumption Warning: Some expected frequencies are less than 5. Results may be inaccurate.")

        #Determine significance
        result_text = "Significant (Reject H0)" if p_val < alpha else "Not Significant (Fail to Reject H0)"

        logger.info(f"Test Complete. P-value: {p_val:.5f} -> Result: {result_text}")

        #Return structured results
        return {
            "chi2_statistic": chi2,
            "p_value": p_val,
            "degrees_of_freedom": dof,
            "significant": p_val < alpha,
            "interpretation": result_text
        }

    #Exception Handling

    except KeyError as e:
        logger.error(f"Column selection error: {e}")

    except TypeError as e:
        logger.error(f"Data type error: {e}")

    except AssertionError as e:
        logger.error(f"Assertion failed: {e}")

    except Exception as e:
        # exception() logs the full traceback for debugging
        logger.exception(f"Critical error during Chi-Square test: {e}")

    return None