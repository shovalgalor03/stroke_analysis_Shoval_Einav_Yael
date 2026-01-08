import pandas as pd
import os
# from src.logger import setup_logger # (Assuming this import works for you)

# logger = setup_logger("Data_Loading") # (Assuming logger is set up)

# --- Main Execution Block ---

# 1. Define the dynamic path to the dataset file
# 'os.path.dirname(__file__)' gets the folder where THIS script is located.
# 'os.path.join' combines it with the filename correctly for both Windows and Mac.
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'healthcare-dataset-stroke-data.csv')

# Note: If the CSV is in a 'data' subfolder, use this instead:
# file_path = os.path.join(current_dir, 'data', 'healthcare-dataset-stroke-data.csv')

# 2. Call the function to load the dataset
# (Make sure 'load_dataset' is defined or imported above)
df = load_dataset(file_path)

# 3. Check if the dataset was loaded successfully
if df is not None:
    print("Process finished successfully.")
    print(f"Loaded file from: {file_path}") # Good for debugging paths
    print(df.head())
else:
    print("Failed to load the dataset.")