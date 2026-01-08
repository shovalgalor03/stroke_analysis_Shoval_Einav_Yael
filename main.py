import os
import pandas as pd

# Imports based on your project structure:
from src.logger import setup_logger           
from src.load_csv import load_dataset         
from src.column_to_numeric import safe_convert_to_numeric 
from src.convert_continuous_to_categorical import convert_continuous_to_categorical
from src.create_composite_variable import create_composite_variable

# Initialize the main logger
logger = setup_logger("Main_Pipeline")

def main():
    logger.info("=== Starting Stroke Prediction Pipeline ===")
    
    # --- Construct File Path ---
    # Using relative path: looks inside the 'stroke_df' folder relative to this script
    file_path = os.path.join("stroke_df", "healthcare-dataset-stroke-data.csv")
    
    # Verify file existence to prevent immediate crash
    if not os.path.exists(file_path):
        logger.critical(f"File not found: {file_path}")
        print("Please make sure the folder 'stroke_df' exists and contains the CSV file.")
        return

    try:
        # --- Step 1: Data Loading ---
        df = load_dataset(file_path)
        
        # Stop execution if loading failed
        if df is None:
            return 

        # --- Step 2: Cleaning and Conversions ---
        
        # Convert specific columns to numeric (e.g., BMI)
        df = safe_convert_to_numeric(df, 'bmi') 
        
        # Create categorical flags (High BMI, High Glucose)
        df = convert_continuous_to_categorical(df)
        
        # Create the composite risk variable (Group assignment)
        df = create_composite_variable(df)

        # --- Completion ---
        logger.info("Pipeline finished successfully.")
        
        # Brief preview to verify success
        print(df.head()) 

    except Exception as e:
        logger.critical(f"Pipeline crashed! Reason: {e}")

if __name__ == "__main__":
    main()