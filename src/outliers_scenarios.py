import pandas as pd
from src.logger import setup_logger

# --- Imports needed for the workflow ---
from src.outliers import remove_outliers_iqr
from src.transformers import convert_continuous_to_categorical, create_composite_variable
from src.relative_risk_analysis import run_full_analysis_pipeline
from src.visualizations import plot_results_table

# Initialize Logger for this module
logger = setup_logger("Scenario_Manager")

def run_scenario(df_base, remove_bmi, remove_glucose, scenario_title):
    """
    Orchestrator function:
    Creates a temporary copy of data, applies specific cleaning logic,
    calculates risk, and generates specific plots.
    """
    logger.info(f"--- Processing Scenario: {scenario_title} ---")
    
    # 1. Work on a fresh copy so we don't mess up the main dataframe
    df_temp = df_base.copy()
    
    # 2. Conditional Outlier Removal
    if remove_bmi:
        df_temp = remove_outliers_iqr(df_temp, 'bmi', threshold=1.5)
    if remove_glucose:
        df_temp = remove_outliers_iqr(df_temp, 'avg_glucose_level', threshold=1.5)
        
    # 3. Feature Engineering (Must run again on the cleaned data)
    df_temp = convert_continuous_to_categorical(df_temp)
    df_temp = create_composite_variable(df_temp)
    
    # 4. Analysis & Visualization
    if 'risk_group' in df_temp.columns:
        # Calculate Relative Risk
        results_df = run_full_analysis_pipeline(df_temp)
        
        if not results_df.empty:
            # Generate the specific Table 
            plot_results_table(results_df, title=scenario_title)
    
        else:
            logger.warning(f"No significant results found for {scenario_title}")
    else:
        logger.error(f"Critical: Failed to create risk groups for {scenario_title}")