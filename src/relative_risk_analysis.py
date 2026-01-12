import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from src.logger import setup_logger

logger = setup_logger("rr_analysis")

def _calculate_statistics(exposed_sick, exposed_healthy, control_sick, control_healthy):
    """
    Internal helper function: Performs only the mathematical calculations.
    Accepts counts of sick/healthy patients and returns: RR, CI, and P-Value.
    """
    # Total people in each group
    total_exposed = exposed_sick + exposed_healthy
    total_control = control_sick + control_healthy

    # 1. Calculate Relative Risk (RR)
    risk_exposed = exposed_sick / total_exposed
    risk_control = control_sick / total_control

    if risk_control == 0:
        rr = np.inf  # Infinity (division by zero is impossible)
    else:
        rr = risk_exposed / risk_control

    # 2. Calculate Confidence Interval (CI)
    try:
        # Standard Error (SE) formula
        se_term_exposed = (1/(exposed_sick + 1e-9)) - (1/(total_exposed + 1e-9))
        se_term_control = (1/(control_sick + 1e-9)) - (1/(total_control + 1e-9))
        se = np.sqrt(se_term_exposed + se_term_control)
        
        # Interval bounds
        ci_lower = np.exp(np.log(rr + 1e-9) - 1.96 * se)
        ci_upper = np.exp(np.log(rr + 1e-9) + 1.96 * se)
    except Exception:
        ci_lower, ci_upper = np.nan, np.nan

    # 3. Calculate P-value (Chi-Square Test)
    obs = np.array([[exposed_sick, exposed_healthy], [control_sick, control_healthy]])
    chi2, p_val, dof, expected = chi2_contingency(obs, correction=False)

    return rr, ci_lower, ci_upper, p_val


def calculate_relative_risk(df, exposed_group, control_group, outcome_col='stroke', group_col='risk_group'):
    """
    Main function: Handles data preparation, validation, and logging.
    Delegates the math to the helper function.
    """
    logger.info(f"Starting analysis: Comparing '{exposed_group}' vs '{control_group}'")

    try:
        # --- 1. Input Validation ---
        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found.")
        
        # Check that groups exist in the data
        existing_groups = df[group_col].unique()
        if exposed_group not in existing_groups or control_group not in existing_groups:
            logger.warning(f"Groups not found. Available: {existing_groups}")
            return None

        # --- 2. Filter Data ---
        subset = df[df[group_col].isin([exposed_group, control_group])]
        
        # Count cases in Exposed Group
        exp_df = subset[subset[group_col] == exposed_group]
        exposed_sick = exp_df[outcome_col].sum()              # Sick (1)
        exposed_healthy = len(exp_df) - exposed_sick          # Healthy (0)

        # Count cases in Control Group
        ctrl_df = subset[subset[group_col] == control_group]
        control_sick = ctrl_df[outcome_col].sum()             # Sick (1)
        control_healthy = len(ctrl_df) - control_sick         # Healthy (0)

        # Check for empty groups
        if (exposed_sick + exposed_healthy) == 0 or (control_sick + control_healthy) == 0:
            logger.error("Cannot calculate RR: One group is empty.")
            return None

        # --- 3. Send to calculation (Helper function) ---
        rr, ci_lower, ci_upper, p_val = _calculate_statistics(
            exposed_sick, exposed_healthy, control_sick, control_healthy
        )

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
        logger.error(f"Error in analysis: {e}")
        raise e


def run_full_analysis_pipeline(df):
    """
    Runs comparisons for all groups and saves the final conclusion.
    """
    logger.info("Running full analysis pipeline...")
    
    comparisons = ['both_high', 'bmi_only', 'glucose_only']
    control = 'neither'
    all_results = []
    
    for group in comparisons:
        # Check if the group exists in the table before sending
        if group in df['risk_group'].values:
            res = calculate_relative_risk(df, group, control)
            if res:
                all_results.append(res)
        else:
            logger.warning(f"Group '{group}' not found. Skipping.")
            
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
        logger.warning("No results to analyze.")

    return results_df