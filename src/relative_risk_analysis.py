import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from src.logger import setup_logger

logger = setup_logger("rr_analysis")

def calculate_statistics(exposed_sick, exposed_healthy, control_sick, control_healthy):
    """
    Internal helper function: Performs only the mathematical calculations.
    Accepts counts of sick/healthy patients and returns: RR, CI, and P-Value.
    """
    # --- SAFETY CHECK 1: Sanity Assertions ---
    # Ensure counts are non-negative (medically impossible).
    assert exposed_sick >= 0 and exposed_healthy >= 0, "Exposed counts cannot be negative"
    assert control_sick >= 0 and control_healthy >= 0, "Control counts cannot be negative"

    # Total people in each group
    total_exposed = exposed_sick + exposed_healthy
    total_control = control_sick + control_healthy

    # --- SAFETY CHECK 2: Empty Groups ---
    # Although checked externally, this internal check prevents mathematical crashes.
    if total_exposed == 0 or total_control == 0:
        return np.nan, np.nan, np.nan, np.nan

    # 1. Calculate Relative Risk (RR)
    risk_exposed = exposed_sick / total_exposed
    risk_control = control_sick / total_control

    if risk_control == 0:
        rr = np.inf  # Infinity (division by zero is impossible)
    else:
        rr = risk_exposed / risk_control

    # 2. Calculate Confidence Interval (CI)
    try:
        # Add epsilon to prevent division by zero in log calculation
        epsilon = 1e-9
        se_term_exposed = (1 / (exposed_sick + epsilon)) - (1 / (total_exposed + epsilon))
        se_term_control = (1 / (control_sick + epsilon)) - (1 / (total_control + epsilon))
        
        # Take absolute value before sqrt to prevent NaN in edge cases
        se = np.sqrt(abs(se_term_exposed + se_term_control))
        
        # Calculate interval bounds
        ci_lower = np.exp(np.log(rr + epsilon) - 1.96 * se)
        ci_upper = np.exp(np.log(rr + epsilon) + 1.96 * se)
    except Exception:
        ci_lower, ci_upper = np.nan, np.nan

    # 3. Calculate P-value (Chi-Square Test)
    obs = np.array([[exposed_sick, exposed_healthy], [control_sick, control_healthy]])
    
    # Check if table is valid for Chi-Square (sum is not zero)
    if np.sum(obs) == 0:
        p_val = 1.0
    else:
        chi2, p_val, dof, expected = chi2_contingency(obs, correction=False)

    return rr, ci_lower, ci_upper, p_val


def calculate_relative_risk(df, exposed_group, control_group, outcome_col='stroke', group_col='risk_group'):
    """
    Main function: Handles data preparation, validation, and logging.
    Delegates the math to the helper function.
    """
    logger.info(f"Starting analysis: Comparing '{exposed_group}' vs '{control_group}'")

    try:
        # --- SAFETY CHECK: Type Checking ---
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input 'df' must be a pandas DataFrame, got {type(df)}")

        # --- 1. Input Validation ---
        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in DataFrame.")
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found.")
        
        # Check that groups exist in the data
        existing_groups = df[group_col].unique()
        if exposed_group not in existing_groups:
            logger.warning(f"Exposed group '{exposed_group}' not found. Skipping.")
            return None
        if control_group not in existing_groups:
            logger.warning(f"Control group '{control_group}' not found. Skipping.")
            return None

        # --- 2. Filter Data ---
        subset = df[df[group_col].isin([exposed_group, control_group])].copy() # Use copy to avoid SettingWithCopyWarning
        
        # Ensure outcome column is numeric (0/1) to prevent summation errors
        if not pd.api.types.is_numeric_dtype(subset[outcome_col]):
             raise TypeError(f"Outcome column '{outcome_col}' must be numeric (0/1).")

        # Count cases in Exposed Group
        exp_df = subset[subset[group_col] == exposed_group]
        exposed_sick = int(exp_df[outcome_col].sum()) # Cast to int for safety
        exposed_healthy = len(exp_df) - exposed_sick

        # Count cases in Control Group
        ctrl_df = subset[subset[group_col] == control_group]
        control_sick = int(ctrl_df[outcome_col].sum()) # Cast to int for safety
        control_healthy = len(ctrl_df) - control_sick

        # --- 3. Send to calculation (Helper function) ---
        rr, ci_lower, ci_upper, p_val = calculate_statistics(
            exposed_sick, exposed_healthy, control_sick, control_healthy)

        # If NaN returned (e.g., due to empty groups), stop here
        if np.isnan(rr):
            logger.warning(f"Skipping {exposed_group}: Not enough data for calculation.")
            return None

        # --- 4. Package Results ---
        results = {
            "comparison": f"{exposed_group} vs {control_group}",
            "RR": round(rr, 4),
            "CI_Lower": round(ci_lower, 4),
            "CI_Upper": round(ci_upper, 4),
            "P_Value": p_val, 
            "Significant_0.05": p_val < 0.05,
            "Exposed_Cases": exposed_sick,
            "Control_Cases": control_sick
        }

        logger.info(f"Analysis Done. RR={results['RR']}, p={results['P_Value']:.4f}")
        return results

    except Exception as e:
        logger.error(f"CRITICAL ERROR in analysis for {exposed_group}: {str(e)}")
        return None  # Return None instead of crashing, allowing the pipeline to continue


def run_full_analysis_pipeline(df):
    """
    Runs comparisons for all groups and saves the final conclusion.
    """
    logger.info("Running full analysis pipeline...")
    
    comparisons = ['both_high', 'bmi_only', 'glucose_only']
    control = 'neither'
    all_results = []
    
    # Preliminary check that DataFrame is not empty
    if df is None or df.empty:
        logger.error("Dataframe is empty or None. Cannot run pipeline.")
        return pd.DataFrame()

    for group in comparisons:
        # Logic to check if group exists is handled inside calculate_relative_risk
        res = calculate_relative_risk(df, group, control)
        if res:
            all_results.append(res)
            
    results_df = pd.DataFrame(all_results)

    # Find the highest risk group
    if not results_df.empty:
        highest_risk = results_df.sort_values(by='RR', ascending=False).iloc[0]
        logger.info("="*50)
        logger.info("SUMMARY - FINAL CONCLUSION:")
        logger.info(f"The group with the HIGHEST Relative Risk is: '{highest_risk['comparison']}'")
        logger.info(f"Risk Factor (RR): {highest_risk['RR']} (CI: {highest_risk['CI_Lower']}-{highest_risk['CI_Upper']})")
        logger.info("="*50)
    else:
        logger.warning("No valid results were generated.")

    # Save as standard CSV 
    results_df.to_csv("final_results.csv", index=False)

    # Save as a formatted text table
    with open("final_results_table.txt", "w") as f:
        f.write(results_df.to_string(index=False))

    logger.info("Saved results to 'final_results.csv' and 'final_results_table.txt'")

    return results_df