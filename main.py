import pandas as pd
from src.logger import setup_logger

# ==========================================
# 1. Imports (Matched EXACTLY to your snippets)
# ==========================================

# A. Loading
from src.load_csv import load_dataset

# B. Cleaning
# Note: Assuming safe_convert_to_numeric is in src/data_cleaning.py or src/column_to_numeric.py
from src.data_cleaning import safe_convert_to_numeric 
#from src.fill_missing import fill_missing_values  # Critical for clustering
from src.remove_columns import remove_columns

# C. Transformation (Feature Engineering)
from src.transformers import convert_continuous_to_categorical, create_composite_variable

# D. Outliers
from src.outliers import remove_outliers_iqr

# E. Statistics
from src.chi_square_analysis import run_chi_square_test
from src.relative_risk_analysis import run_full_analysis_pipeline

# F. Clustering
from src.cluster_analysis import perform_clustering, plot_clusters_pca, plot_risk_analysis

# Initialize Logger
logger = setup_logger("Main_Runner")

def main():
    """
    Main Execution Pipeline.
    Orchestrates the entire analysis flow using the provided modules.
    """
    logger.info(">>> Starting the Stroke Analysis Pipeline... <<<")

    # ---------------------------------------------------------
    # Step 1: Load Data
    # ---------------------------------------------------------
    file_path = "stroke_df/healthcare-dataset-stroke-data.csv"
    df = load_dataset(file_path)
    
    if df is None:
        logger.error("CRITICAL: Exiting due to data loading failure.")
        return

    # ---------------------------------------------------------
    # Step 2: Data Cleaning & Prep
    # ---------------------------------------------------------
    logger.info("--- Phase 1: Data Cleaning ---")
    
    # 1. Remove ID (irrelevant for model/stats)
    df = remove_columns(df, columns_to_remove=['id'])

    # 2. Convert types (using your safe_convert_to_numeric)
    # We must convert these before filling missing values
    for col in ['bmi', 'avg_glucose_level', 'age']:
        df = safe_convert_to_numeric(df, col)

    # 3. Fill Missing Values
    # (Essential because K-Means and some stats fail with NaNs)
    try:
        df = fill_missing_values(df, strategy='median')
    except NameError:
        logger.warning("fill_missing_values not found. Skipping (Clustering might fail).")

    # ---------------------------------------------------------
    # Step 3: Outlier Detection
    # ---------------------------------------------------------
    logger.info("--- Phase 2: Outlier Detection ---")
    
    # Using your remove_outliers_iqr function
    df = remove_outliers_iqr(df, 'bmi', threshold=2.0)
    df = remove_outliers_iqr(df, 'avg_glucose_level', threshold=1.5)

    # ---------------------------------------------------------
    # Step 4: Feature Engineering
    # ---------------------------------------------------------
    logger.info("--- Phase 3: Feature Engineering ---")

    # 1. Create Flags (bmi_high, glucose_high) - Calls convert_continuous_to_categorical
    df = convert_continuous_to_categorical(df)

    # 2. Create Risk Groups - Calls create_composite_variable
    df = create_composite_variable(df)

    # ---------------------------------------------------------
    # Step 5: Statistical Analysis
    # ---------------------------------------------------------
    logger.info("--- Phase 4: Statistical Analysis ---")

    if 'risk_group' in df.columns:
        # 1. Chi-Square (Independence)
        # Calls run_chi_square_test -> calls create_contingency_table internally
        run_chi_square_test(df, independent_var='risk_group', dependent_var='stroke')

        # 2. Relative Risk (Hypothesis Testing)
        # Calls run_full_analysis_pipeline -> calls calculate_relative_risk internally
        rr_results = run_full_analysis_pipeline(df)
        
        # Print summary to terminal
        if not rr_results.empty:
            print("\n" + "="*50)
            print(" FINAL RELATIVE RISK RESULTS ")
            print("="*50)
            print(rr_results[['comparison', 'RR', 'P_Value', 'Significant_0.05']])
            print("="*50 + "\n")
    else:
        logger.error("Skipping statistical analysis: 'risk_group' column missing.")

    # ---------------------------------------------------------
    # Step 6: Cluster Analysis
    # ---------------------------------------------------------
    logger.info("--- Phase 5: Cluster Analysis ---")
    
    try:
        # 1. Run K-Means - Calls perform_clustering
        df_clustered = perform_clustering(df, n_clusters=3)
        
        # 2. Visualize - Calls plot_clusters_pca
        plot_clusters_pca(df_clustered)
        
        # 3. Analyze Risk per Cluster - Calls plot_risk_analysis
        cluster_summary = df_clustered.groupby('cluster')['stroke'].mean().reset_index()
        cluster_summary.columns = ['cluster', 'stroke_risk_%']
        cluster_summary['stroke_risk_%'] *= 100
        
        plot_risk_analysis(cluster_summary)
        
    except Exception as e:
        logger.error(f"Clustering pipeline failed: {e}")

    logger.info("Pipeline Finished Successfully. ")

if __name__ == "__main__":
    main()