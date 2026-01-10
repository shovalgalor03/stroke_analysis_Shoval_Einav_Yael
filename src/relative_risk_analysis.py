import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from src.logger import setup_logger

# Initialize a specific logger for this module
logger = setup_logger("rr_analysis")

def calculate_relative_risk(df, exposed_group, control_group, outcome_col='stroke', group_col='risk_group'):
    """
    Calculates Relative Risk (RR), Confidence Interval (CI), and P-value (Chi-Square).
    Arguments are passed without strict type hinting for cleaner code.
    """
    logger.info(f"Starting analysis: Comparing '{exposed_group}' vs '{control_group}'")

    try:
        # --- 1. Input Validation ---
        # Check if the grouping column exists
        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in DataFrame.")
            
        # Check if groups exist in the data
        unique_groups = df[group_col].unique()
        if exposed_group not in unique_groups or control_group not in unique_groups:
             logger.warning(f"One of the groups was not found. Available: {unique_groups}")
             return None

        # --- 2. Filter Data ---
        # Keep only rows belonging to the two relevant groups
        subset = df[df[group_col].isin([exposed_group, control_group])].copy()
        
        # --- 3. Construct 2x2 Table ---
        # Calculate counts for Exposed Group
        # A = Exposed + Sick (Stroke), B = Exposed + Healthy
        exposed_data = subset[subset[group_col] == exposed_group]
        a = exposed_data[outcome_col].sum()
        b = len(exposed_data) - a
        
        # Calculate counts for Control Group
        # C = Control + Sick (Stroke), D = Control + Healthy
        control_data = subset[subset[group_col] == control_group]
        c = control_data[outcome_col].sum()
        d = len(control_data) - c

        total_exposed = a + b
        total_control = c + d

        # Safety check for empty groups
        if total_exposed == 0 or total_control == 0:
            logger.error("Cannot calculate RR: One group is empty.")
            return None

        # --- 4. Calculate Relative Risk (RR) ---
        risk_exposed = a / total_exposed
        risk_control = c / total_control

        if risk_control == 0:
            rr = np.inf  # Mathematical infinity if control risk is 0
        else:
            rr = risk_exposed / risk_control

        # --- 5. Calculate 95% Confidence Interval (CI) ---
        # Using the Log Method for RR
        try:
            se = np.sqrt((1/(a + 1e-9)) + (1/(c + 1e-9)) - (1/(total_exposed + 1e-9)) - (1/(total_control + 1e-9)))
            ci_lower = np.exp(np.log(rr + 1e-9) - 1.96 * se)
            ci_upper = np.exp(np.log(rr + 1e-9) + 1.96 * se)
        except Exception:
            # Fallback if math error occurs
            ci_lower, ci_upper = np.nan, np.nan

        # --- 6. Calculate P-value (Chi-Square Test) ---
        obs = np.array([[a, b], [c, d]])
        chi2, p_val, dof, expected = chi2_contingency(obs, correction=False)

        # --- 7. Package Results ---
        results = {
            "comparison": f"{exposed_group} vs {control_group}",
            "RR": round(rr, 4),
            "CI_Lower": round(ci_lower, 4),
            "CI_Upper": round(ci_upper, 4),
            "P_Value": p_val, 
            "Significant_0.05": p_val < 0.05,
            "Exposed_Cases": a,
            "Exposed_Total": total_exposed,
            "Control_Cases": c,
            "Control_Total": total_control
        }

        logger.info(f"Analysis Done. RR={results['RR']}, p={results['P_Value']:.4f}")
        return results

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise e

def run_full_analysis_pipeline(df):
    """
    Runs the comparisons for all groups against 'neither' and logs the conclusion.
    """
    logger.info("Running full analysis pipeline...")
    
    # Define the groups to compare
    comparisons = ['both_high', 'bmi_only', 'glucose_only']
    control = 'neither'
    
    all_results = []
    
    # Loop through each risk group
    for group in comparisons:
        if group in df['risk_group'].values:
            
            # --- THE CLEAN LINE ---
            # Python automatically maps: 1st->df, 2nd->exposed, 3rd->control
            res = calculate_relative_risk(df, group, control)
            
            if res:
                all_results.append(res)
        else:
            logger.warning(f"Group '{group}' not found. Skipping.")
            
    # Convert list to DataFrame
    results_df = pd.DataFrame(all_results)

    # --- NEW: Log Final Conclusion (Added based on request) ---
    if not results_df.empty:
        # Sort by RR descending (highest first)
        sorted_df = results_df.sort_values(by='RR', ascending=False)
        
        # Get the top row
        highest_risk = sorted_df.iloc[0]
        
        # Log the conclusion nicely
        logger.info("="*50) # to create a seperation
        logger.info("SUMMARY - FINAL CONCLUSION:")
        logger.info(f"The group with the HIGHEST Relative Risk is: '{highest_risk['comparison']}'")
        logger.info(f"Risk Factor (RR): {highest_risk['RR']} (CI: {highest_risk['CI_Lower']}-{highest_risk['CI_Upper']})")
        logger.info("="*50)
    else:
        logger.warning("No results to analyze.")

    return results_df