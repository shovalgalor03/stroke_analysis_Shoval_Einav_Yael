import pandas as pd
from src.logger import setup_logger

from src.outliers import remove_outliers_iqr
from src.transformers import convert_continuous_to_categorical, create_composite_variable
from src.relative_risk_analysis import run_full_analysis_pipeline
from src.visualizations import plot_results_table

# Initialize Logger for this module
logger = setup_logger("glucose_outliers_scenario")
        
def run_scenario(df_base, remove_glucose, scenario_title):
    """
    Executes function:
    1. Identifies outliers.
    2. Checks which 'Risk Group' those outliers belonged to (Specific Count).
    3. Runs analysis on cleaned data.
    4. Updates the table with specific outlier counts per comparison row.
    """
    logger.info(f"--- Processing Scenario: {scenario_title} ---")
    
    # 1. Capture original indices to track who gets dropped
    original_indices = df_base.index
    
    # 2. Create the Cleaned DataFrame
    df_clean = df_base.copy()
    if remove_glucose:
        df_clean = remove_outliers_iqr(df_clean, 'avg_glucose_level', threshold=1.5)
        
    # 3. Identify exactly WHO was dropped
    remaining_indices = df_clean.index
    dropped_indices = original_indices.difference(remaining_indices)
    
    # 4. Analyze the dropped rows to see which group they would have been in
    # Extract dropped rows from the original base
    df_dropped_rows = df_base.loc[dropped_indices].copy()
    
    # Classify rows to determine if they are 'both_high', 'neither'
    if not df_dropped_rows.empty:
        df_dropped_rows = convert_continuous_to_categorical(df_dropped_rows)
        df_dropped_rows = create_composite_variable(df_dropped_rows)
        # Count outliers per group 
        outlier_counts = df_dropped_rows['risk_group'].value_counts().to_dict()
    else:
        outlier_counts = {}

    # 5. Process the CLEAN data for the actual report
    df_clean = convert_continuous_to_categorical(df_clean)
    df_clean = create_composite_variable(df_clean)
    
    # 6. Analysis & Visualization
    if 'risk_group' in df_clean.columns:
        results_df = run_full_analysis_pipeline(df_clean)
        
        if not results_df.empty:
            
            def calculate_specific_drop(row):
                # It needs to be split to find the two groups.
                try:
                    # split string "group A vs group B"
                    parts = row['comparison'].split(' vs ')
                    group_a = parts[0]
                    group_b = parts[1]
                    
                    # Sum the outliers from Group A and Group B
                    count_a = outlier_counts.get(group_a, 0)
                    count_b = outlier_counts.get(group_b, 0)
                    return count_a + count_b
                except:
                    return 0

            # Apply the calculation to every row
            results_df['Outliers_Removed'] = results_df.apply(calculate_specific_drop, axis=1)
            
            # Generate Table
            plot_results_table(results_df, title=scenario_title)
    
        else:
            logger.warning(f"No significant results found for {scenario_title}")
    else:
        logger.error(f"Critical: Failed to create risk groups for {scenario_title}")




 
