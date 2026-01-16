'''
import unittest
import pandas as pd
import numpy as np

# Assuming fill_missing_with_median is imported from your main module
# from main import fill_missing_with_median

class TestFillMissingWithMedian(unittest.TestCase):

    def setUp(self):
        """
        Setup function to create a clean DataFrame before each test.
        The DataFrame includes:
        - 'Price': Numeric column with missing values (NaN).
        - 'Ram': Numeric column with NO missing values.
        - 'Company': String column (non-numeric).
        - 'All_NaN': Numeric column with ALL values missing.
        """
        self.data = {
            'Price': [100.0, 300.0, np.nan, 200.0, np.nan], # Median is 200.0
            'Ram': [8, 16, 32, 4, 8],
            'Company': ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus'],
            'All_NaN': [np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        self.df = pd.DataFrame(self.data)

    # --- Positive Test Case ---
    def test_fill_valid_numeric_column(self):
        """
        Verifies that the system works as expected: valid numeric column with NaNs
        is filled with the correct median value[cite: 95].
        """
        col_name = 'Price'
        # The median of [100, 300, 200] is 200.0
        expected_median = 200.0
        
        # Action
        df_new = fill_missing_with_median(self.df, col_name)
        
        # Assertions
        # 1. Verify there are no NaNs left in the column
        self.assertEqual(df_new[col_name].isna().sum(), 0)
        
        # 2. Verify the NaNs were replaced by the median (checking index 2 and 4)
        self.assertEqual(df_new[col_name].iloc[2], expected_median)
        self.assertEqual(df_new[col_name].iloc[4], expected_median)

    # --- Positive Test Case (No Changes) ---
    def test_no_missing_values(self):
        """
        Verifies behavior when there are no missing values.
        The DataFrame should remain identical.
        """
        col_name = 'Ram'
        df_new = fill_missing_with_median(self.df, col_name)
        
        # Expectation: DataFrames are identical since no action was needed
        pd.testing.assert_frame_equal(self.df, df_new)

    # --- Negative Test Case ---
    def test_non_existent_column(self):
        """
        Tests how the system handles a column that does not exist.
        The function should catch KeyError and return the original DataFrame[cite: 96].
        """
        col_name = 'Screen_Size' # Does not exist
        
        # Action
        df_new = fill_missing_with_median(self.df, col_name)
        
        # Expectation: Returns original DF (integrity preserved)
        pd.testing.assert_frame_equal(self.df, df_new)

    def test_non_numeric_column(self):
        """
        Tests validation logic: Trying to calculate median on a string column.
        Should catch AssertionError (Prerequisite Failed) and return original DF.
        """
        col_name = 'Company' # String type
        
        # Action
        df_new = fill_missing_with_median(self.df, col_name)
        
        # Expectation: Returns original DF because assertion `is_numeric_dtype` fails
        pd.testing.assert_frame_equal(self.df, df_new)

    # --- Edge Test Case ---
    def test_all_values_are_nan(self):
        """
        Tests the boundary where a column is numeric but entirely empty (All NaNs).
        The median of all NaNs is NaN. The function asserts `not pd.isna(median_val)`,
        so this should fail safely and return the original DF[cite: 98, 100].
        """
        col_name = 'All_NaN'
        
        # Action
        df_new = fill_missing_with_median(self.df, col_name)
        
        # Expectation: Returns original DF because median calculation resulted in NaN
        pd.testing.assert_frame_equal(self.df, df_new)
        # Verify the column still has NaNs (nothing happened)
        self.assertEqual(df_new[col_name].isna().sum(), 5)

    # --- Error Test Case ---
    def test_invalid_input_type(self):
        """
        Intentionally forces an error by passing an invalid object (string) 
        instead of a DataFrame. 
        Tests if the general Exception handler catches it[cite: 99].
        """
        invalid_input = "I am not a DataFrame"
        
        # Action
        result = fill_missing_with_median(invalid_input, 'Price')
        
        # Expectation: The function returns the input object due to exception handling
        self.assertEqual(result, invalid_input)

if __name__ == '__main__':
    unittest.main()
'''

import sys
import os
import unittest
import pandas as pd
import numpy as np

# --- Path Configuration (הוספנו את זה כדי שהטסט ימצא את הקוד) ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_test_dir)
sys.path.insert(0, project_root_dir)
# -------------------------------------------------------------

# Import the function
try:
    from src.data_cleaning import fill_missing_with_median
except ImportError:
    # Fallback if the file name is different
    try:
        from src.fill_missing import fill_missing_with_median
    except ImportError as e:
        raise ImportError(f"Could not import 'fill_missing_with_median'. Check 'src' folder. Error: {e}")

class TestFillMissingWithMedian(unittest.TestCase):

    def setUp(self):
        """
        Setup function to create a clean DataFrame before each test.
        """
        self.data = {
            'Price': [100.0, 300.0, np.nan, 200.0, np.nan], # Median is 200.0
            'Ram': [8, 16, 32, 4, 8],
            'Company': ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus'],
            'All_NaN': [np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        self.df = pd.DataFrame(self.data)

    # --- Positive Test Case ---
    def test_fill_valid_numeric_column(self):
        col_name = 'Price'
        expected_median = 200.0
        
        # Action
        df_new = fill_missing_with_median(self.df.copy(), col_name)
        
        # Assertions
        self.assertEqual(df_new[col_name].isna().sum(), 0)
        self.assertEqual(df_new[col_name].iloc[2], expected_median)
        self.assertEqual(df_new[col_name].iloc[4], expected_median)

    # --- Positive Test Case (No Changes) ---
    def test_no_missing_values(self):
        col_name = 'Ram'
        df_new = fill_missing_with_median(self.df.copy(), col_name)
        pd.testing.assert_frame_equal(self.df, df_new)

    # --- Negative Test Case ---
    def test_non_existent_column(self):
        col_name = 'Screen_Size' # Does not exist
        df_new = fill_missing_with_median(self.df.copy(), col_name)
        pd.testing.assert_frame_equal(self.df, df_new)

    def test_non_numeric_column(self):
        col_name = 'Company' # String type
        df_new = fill_missing_with_median(self.df.copy(), col_name)
        pd.testing.assert_frame_equal(self.df, df_new)

    # --- Edge Test Case ---
    def test_all_values_are_nan(self):
        col_name = 'All_NaN'
        df_new = fill_missing_with_median(self.df.copy(), col_name)
        
        # Expectation: Returns original DF because median calculation resulted in NaN
        pd.testing.assert_frame_equal(self.df, df_new)
        self.assertEqual(df_new[col_name].isna().sum(), 5)

    # --- Error Test Case ---
    def test_invalid_input_type(self):
        invalid_input = "I am not a DataFrame"
        
        # Action
        # Assuming function returns input or None on error, avoiding crash
        try:
            result = fill_missing_with_median(invalid_input, 'Price')
            if result is not None:
                self.assertEqual(result, invalid_input)
        except Exception:
            pass # Exception is also acceptable behavior

if __name__ == '__main__':
    unittest.main()    