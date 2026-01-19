import pandas as pd
import os
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
from src.visualizations import plot_all_visualizations

# G. Clustering
from src.cluster_analysis import find_optimal_k,perform_clustering, plot_clusters_pca, plot_risk_analysis, get_cluster_profiles, plot_cluster_profile_table, plot_stroke_capture_rate, prepare_data, calculate_cluster_risks

# H. Scenario Manager (NEW)
from src.glucose_outliers_scenario import run_scenario

# Initialize Logger
logger = setup_logger("Main_Runner")

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

    # --- Step 3: Generate Outlier Scenario (Table) ---
    # Run Scenario Reports
    logger.info("--- Phase 2: Generating Scenario Report ---")
    run_scenario(df, remove_glucose=True, scenario_title="Without Glucose Outliers")
   
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
    
    output_dir = "clustering_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Configuration for the two requested runs
    scenarios = [
        {"cols": ['stroke'], "name": "with_glucose"},
        {"cols": ['stroke', 'avg_glucose_level'], "name": "no_glucose"}]

    for sc in scenarios:
        suffix = sc["name"]
        logger.info(f"Processing clustering for: {suffix}")
        
        try:
            df_blind = remove_columns(df, columns_to_remove=sc["cols"]) # removing the target variable ('id' column already removed)
            X_scaled = prepare_data(df_blind)
            
            optimal_k = find_optimal_k(X_scaled, 
                                       save_path=os.path.join(output_dir, f"optimal_k_{suffix}.png")) # Automatically find the mathematically optimal number of clusters
            df_clustered = perform_clustering(df_blind, X_scaled, n_clusters=optimal_k) # Execute clustering algorithm using the discovered K

            df_clustered['stroke'] = df['stroke'] # Re-attach 'stroke' diagnosis to validate the clusters

            plot_clusters_pca(X_scaled, df_clustered['cluster'], 
                            save_path=os.path.join(output_dir, f"pca_{suffix}.png")) 
                        
            # Calculate percentage of stroke cases per cluster
            cluster_summary = calculate_cluster_risks(df_clustered)
            plot_risk_analysis(cluster_summary,
                            save_path=os.path.join(output_dir, f"risk_{suffix}.png")) 
            
            plot_stroke_capture_rate(df_clustered,
                                    save_path=os.path.join(output_dir, f"capture_{suffix}.png"))
            
            profile_data = get_cluster_profiles(df_clustered) 
            plot_cluster_profile_table(profile_data, 
                                    save_path=os.path.join(output_dir, f"profile_{suffix}.png"))
            
            logger.info(f"SUCCESS: Clustering section complete with {optimal_k} clusters.")
    
        except Exception as e:
            logger.error(f"Clustering pipeline failed: {e}")

    logger.info(">>> Pipeline Finished Successfully. <<<")

if __name__ == "__main__":
    main()