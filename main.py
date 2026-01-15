import pandas as pd
from src.logger import setup_logger

# A. Loading
from src.load_csv import load_dataset

# B. Cleaning
from src.data_cleaning import convert_to_numeric, fill_missing_with_median, remove_columns 

# C. Transformation (Feature Engineering)
from src.transformers import convert_continuous_to_categorical, create_composite_variable

# D. Outliers
from src.outliers import remove_outliers_iqr

# E. Statistics
from src.chi_square_analysis import run_chi_square_test
from src.relative_risk_analysis import run_full_analysis_pipeline

# F. visualizations
from src.visualizations import plot_all_visualizations, plot_results_table, plot_rr_forest_plot

# G. Clustering
from src.cluster_analysis import find_optimal_k,perform_clustering, plot_clusters_pca, plot_risk_analysis, get_cluster_profiles, plot_cluster_profile_table, plot_stroke_capture_rate

# Initialize Logger
logger = setup_logger("Main_Runner")

def run_scenario(df_base, remove_bmi, remove_glucose, scenario_title):
    """
    Helper function: Creates a TEMPORARY copy of data, removes outliers, 
    and generates the requested table/plot without affecting the main pipeline.
    """
    logger.info(f"--- Generating Report: {scenario_title} ---")
    
    # 1. Work on a fresh copy (Safe!)
    df_temp = df_base.copy()
    
    # 2. Conditional Outlier Removal on the copy
    if remove_bmi:
        df_temp = remove_outliers_iqr(df_temp, 'bmi', threshold=1.5)
    if remove_glucose:
        df_temp = remove_outliers_iqr(df_temp, 'avg_glucose_level', threshold=1.5)
        
    # 3. Feature Engineering on the copy
    df_temp = convert_continuous_to_categorical(df_temp)
    df_temp = create_composite_variable(df_temp)
    
    # 4. Generate Visualizations
    if 'risk_group' in df_temp.columns:
        results_df = run_full_analysis_pipeline(df_temp)
        if not results_df.empty:
            # Create the specific Table and Forest Plot image
            plot_results_table(results_df, title=scenario_title)
            plot_rr_forest_plot(results_df, title_suffix=scenario_title)
        else:
            logger.warning(f"No significant results found for {scenario_title}")

def main():
    """
    Main Execution Pipeline.
    Orchestrates the entire analysis flow using the provided modules.
    """
    logger.info(">>> Starting the Stroke Analysis Pipeline... <<<")

    # --- Step 1: Load Data
    file_path = "stroke_df/healthcare-dataset-stroke-data.csv"
    df = load_dataset(file_path)
    
    if df is None:
        logger.error("CRITICAL: Exiting due to data loading failure.")
        return

    # --- Step 2: Data Cleaning & Prep
    logger.info("--- Phase 1: Data Cleaning ---")
    
    # 1. Remove ID (irrelevant for model/stats)
    df = remove_columns(df, columns_to_remove=['id'])

    # 2. Convert types (using your safe_convert_to_numeric)
    # We must convert these before filling missing values
    for col in ['bmi', 'avg_glucose_level', 'age']:
        df = convert_to_numeric(df, col)

    # 3. Fill Missing Values
    # (Essential because K-Means and some stats fail with NaNs)
    try:
        df = fill_missing_with_median(df,'bmi')
    except NameError:
        logger.warning("fill_missing_with_median not found. Skipping (Clustering might fail).")

   # --- Step 3: Generate Outlier Scenarios (Tables & Plots) ---
    # This block generates the 3 specific tables you asked for.
    # It uses copies, so your main 'df' remains FULL for the rest of the analysis.
    
    logger.info("--- Phase 2: Generating Multi-Scenario Reports ---")
    # Scenario 1: Without BMI Outliers
    run_scenario(df, remove_bmi=True, remove_glucose=False, scenario_title="1. Without BMI Outliers")

    # Scenario 2: Without Glucose Outliers
    run_scenario(df, remove_bmi=False, remove_glucose=True, scenario_title="2. Without Glucose Outliers")

    # Scenario 3: Without Any Outliers
    run_scenario(df, remove_bmi=True, remove_glucose=True, scenario_title="3. Without Any Outliers")

    # --- Step 4: Feature Engineering
    logger.info("--- Phase 3: Feature Engineering ---")
    df = convert_continuous_to_categorical(df) # Create Flags (bmi_high, glucose_high) 
    df = create_composite_variable(df) # Create Risk Groups

    # --- Step 5: Statistical Analysis
    logger.info("--- Phase 4: Statistical Analysis ---")

    if 'risk_group' in df.columns:
        # 1. Chi-Square (Independence)
        run_chi_square_test(df, independent_var='risk_group', dependent_var='stroke') # calls create_contingency_table internally

        # 2. Relative Risk (Hypothesis Testing)
        rr_results = run_full_analysis_pipeline(df) # calls calculate_relative_risk internally
        
        # Print summary to terminal
        if not rr_results.empty:
            print("\n" + "="*50)
            print(" FINAL RELATIVE RISK RESULTS ")
            print("="*50)
            print(rr_results[['comparison', 'RR', 'P_Value', 'Significant_0.05']])
            print("="*50 + "\n")
    else:
        logger.error("Skipping statistical analysis: 'risk_group' column missing.")

    # --- Step 6: Visualizations
    logger.info("Generating visualizations...")
    results_df = run_full_analysis_pipeline(df)
    plot_all_visualizations(df, results_df)

    logger.info("All visualizations have been successfully saved to the project folder.")


    # --- Step 7: Cluster Analysis (Extra)
    logger.info("--- Phase 5: Cluster Analysis ---")

    try:
        df_blind = remove_columns(df, columns_to_remove=['stroke']) # removing the target variable and non-relevant column
        optimal_k = find_optimal_k(df_blind) # Automatically find the mathematically optimal number of clusters
        df_clustered = perform_clustering(df_blind, n_clusters=optimal_k) # Execute clustering algorithm using the discovered K

        plot_clusters_pca(df_clustered) # Visualize the patient segments in 2D space (PCA)
        
        df_clustered['stroke'] = df['stroke'] # Re-attach 'stroke' diagnosis to validate the clusters
        
        profile_data = get_cluster_profiles(df_clustered) 
        plot_cluster_profile_table(profile_data, filename="cluster_profile_table.png")   
             
        # Calculate percentage of stroke cases per cluster
        df_clustered['stroke'] = pd.to_numeric(df_clustered['stroke'], errors='coerce') # Ensure 'stroke' is numeric to avoid calculation errors
        cluster_summary = df_clustered.groupby('cluster')['stroke'].mean().reset_index()
        cluster_summary.columns = ['cluster', 'stroke_risk_%']
        cluster_summary['stroke_risk_%'] *= 100
        
        # Visualization of cluster performance
        plot_stroke_capture_rate(df_clustered)   
        plot_risk_analysis(cluster_summary)
        
        logger.info(f"SUCCESS: Clustering section complete with {optimal_k} clusters.")
   
    except Exception as e:
        logger.error(f"Clustering pipeline failed: {e}")

    logger.info(">>> Pipeline Finished Successfully. <<<")

if __name__ == "__main__":
    main()