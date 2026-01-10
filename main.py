import pandas as pd
import sys
import os

# --- Imports from your src folder ---
from src.logger import setup_logger

# 1. Data Loading
from src.load_csv import load_dataset

# 2. Data Cleaning & Preparation (Yael's part)
# (Make sure function names match exactly what is written in the files)
from src.column_to_numeric import safe_convert_to_numeric
from src.fill_missing import fill_missing_values 
# from src.convert_continuous_to_categorical import convert_continuous_to_categorical # Optional if needed separately
from src.create_composite_variable import create_risk_groups

# 3. Statistical Analysis (Einav & Shoval's part)
from src.chi_square_analysis import run_chi_square_test
from src.relative_risk_analysis import run_full_analysis_pipeline

# Initialize the main logger
logger = setup_logger("Main_Pipeline")

def main():
    """
    Main execution function.
    Runs the full data analysis pipeline: Load -> Clean -> Enrich -> Analyze.
    """
    logger.info("==========================================")
    logger.info("   STARTING STROKE ANALYSIS PIPELINE      ")
    logger.info("==========================================")

    # ---------------------------------------------------------
    # STEP 1: Load Data
    # ---------------------------------------------------------
    file_path = "stroke_df/healthcare-dataset-stroke-data.csv"
    logger.info(f"Step 1: Loading dataset from {file_path}...")
    
    df = load_dataset(file_path)

    # Safety check: If data didn't load, stop everything.
    if df is None or df.empty:
        logger.critical("Data loading failed or dataframe is empty. Exiting.")
        return

    # ---------------------------------------------------------
    # STEP 2: Data Cleaning & Preprocessing
    # ---------------------------------------------------------
    logger.info("Step 2: Cleaning and Preprocessing Data...")

    # A. Convert columns to numeric (handling errors/strings)
    try:
        df = safe_convert_to_numeric(df)
        logger.info(" - Converted relevant columns to numeric.")
    except Exception as e:
        logger.error(f"Error in convert_to_numeric: {e}")

    # B. Fill missing values (NaN -> Mean/Mode)
    try:
        df = fill_missing_values(df)
        logger.info(" - Filled missing values.")
    except Exception as e:
        logger.error(f"Error in fill_missing_values: {e}")

    # C. Create new variables (Risk Groups)
    # This is critical for the Relative Risk analysis later
    try:
        df = create_risk_groups(df)
        logger.info(" - Created 'risk_group' composite variable.")
    except Exception as e:
        logger.error(f"Error in create_risk_groups: {e}")

    # ---------------------------------------------------------
    # STEP 3: Statistical Analysis
    # ---------------------------------------------------------
    logger.info("Step 3: Running Statistical Analysis...")

    # Analysis A: Chi-Square Test (Einav's Module)
    # Checks for general association between risk groups and stroke
    logger.info("--> Running Chi-Square Test (General Association)...")
    try:
        run_chi_square_test(df, independent_var='risk_group', dependent_var='stroke')
    except Exception as e:
        logger.error(f"Error in Chi-Square Test: {e}")

    # Analysis B: Relative Risk Analysis (Shoval's Module)
    # Checks specific risk for each group compared to 'neither'
    logger.info("--> Running Relative Risk Analysis (Specific Group Comparisons)...")
    try:
        results_df = run_full_analysis_pipeline(df)
        
        # Optional: Print a sneak peek of the results to the terminal
        if results_df is not None and not results_df.empty:
            print("\n--- Final Results Preview ---")
            print(results_df[['comparison', 'RR', 'P_Value', 'Significant_0.05']])
            print("-----------------------------\n")
            
    except Exception as e:
        logger.error(f"Error in Relative Risk Analysis: {e}")

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------
    logger.info("==========================================")
    logger.info("   ANALYSIS COMPLETE. CHECK LOGS.         ")
    logger.info("==========================================")

if __name__ == "__main__":
    main()