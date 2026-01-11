import pandas as pd
from src.logger import setup_logger

# --- Imports (Matched exactly to your file names) ---
from src.load_csv import load_dataset 

# 1. Convert to Numeric + fill missing
from src.data_cleaning import convert_to_numeric, fill_missing_with_median 

# 2. Create Categorical Columns + Create Composite Variable
from src.transformers import convert_continuous_to_categorical, create_composite_variable  

# 3. Statistical Analysis
from src.chi_square_analysis import run_chi_square_test
from src.relative_risk_analysis import run_full_analysis_pipeline

logger = setup_logger("Main_Runner")

def main():
    logger.info("Starting the Stroke Analysis Pipeline...")

    # --- Step 1: Load Data ---
    file_path = "stroke_df/healthcare-dataset-stroke-data.csv" 
    df = load_dataset(file_path)
    
    if df is None:
        logger.error("Exiting due to data loading failure.")
        return

    # --- Step 2: Cleaning and Preparation ---
    logger.info("--- Cleaning Data ---")
    
    # Converting columns to numeric (sending 'col_name' as required)
    try:
        # We call the function twice, once for each relevant column
        df = safe_convert_to_numeric(df, 'bmi')
        df = safe_convert_to_numeric(df, 'avg_glucose_level')
        logger.info(" - Converted columns to numeric.")
    except Exception as e:
        logger.error(f"Error in safe_convert_to_numeric: {e}")

    # Filling missing values (sending column name)
    try:
        df = fill_missing_with_median(df, 'bmi') 
        logger.info(" - Filled missing values in 'bmi'.")
    except Exception as e:
        logger.error(f"Error in fill_missing_with_median: {e}")

    # Critical Step: Create categorical columns (bmi_high, etc.)
    # The next function (composite) requires these columns to exist
    try:
        df = convert_continuous_to_categorical(df)
        logger.info(" - Created categorical columns (bmi_high, glucose_high).")
    except Exception as e:
        logger.error(f"Error in convert_continuous_to_categorical: {e}")

    # Create Composite Variable (using the correct function name)
    try:
        df = create_composite_variable(df)
        logger.info(" - Created 'risk_group' variable.")
    except Exception as e:
        logger.error(f"Error in create_composite_variable: {e}")

    # --- Step 3: Statistical Analysis ---
    logger.info("--- Running Analysis ---")
    
    if 'risk_group' in df.columns:
        # 1. Chi-Square Test
        logger.info("Running Chi-Square...")
        run_chi_square_test(df, 'risk_group', 'stroke')

        # 2. Relative Risk Analysis
        logger.info("Running Relative Risk...")
        results = run_full_analysis_pipeline(df)
        
        # Print summary to terminal
        if results is not None:
            print("\n" + "="*40)
            print(" FINAL RESULTS PREVIEW ")
            print("="*40)
            print(results[['comparison', 'RR', 'P_Value', 'Significant_0.05']])
            print("\n")
    else:
        logger.error("CRITICAL: 'risk_group' column missing. Cannot run analysis.")

if __name__ == "__main__":
    main()