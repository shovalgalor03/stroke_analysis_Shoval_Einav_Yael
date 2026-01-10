import unittest
import pandas as pd
import numpy as np

# Assuming remove_columns is imported from your main module
# from main import remove_columns 

class TestRemoveColumns(unittest.TestCase):

    def setUp(self):
        """
        Setup function to create a clean DataFrame before each test.
        This ensures isolation between tests[cite: 143].
        """
        self.data = {
            'laptop_ID': [1, 2, 3],
            'Company': ['Apple', 'HP', 'Dell'],
            'Price': [1000, 500, 700],
            'Ram': [8, 16, 32]
        }
        self.df = pd.DataFrame(self.data)

    # --- Positive Test Case [cite: 95] ---
    def test_remove_existing_columns(self):
        """
        Verifies that valid columns are successfully removed from the DataFrame.
        """
        columns_to_remove = ['laptop_ID', 'Company']
        
        # Action
        df_new = remove_columns(self.df, columns_to_remove)
        
        # Assertions [cite: 133]
        self.assertNotIn('laptop_ID', df_new.columns)
        self.assertNotIn('Company', df_new.columns)
        self.assertIn('Price', df_new.columns) # Ensure other columns remain
        self.assertEqual(df_new.shape[1], 2)   # Verify new shape

    # --- Negative Test Case [cite: 96] ---
    def test_remove_non_existent_column(self):
        """
        Tests how the system handles columns that do not exist.
        The function should drop valid columns and ignore missing ones without crashing.
        """
        # 'Ram' exists, 'Screen' does not
        columns_to_remove = ['Screen', 'Ram']
        
        df_new = remove_columns(self.df, columns_to_remove)
        
        # 'Ram' should be removed
        self.assertNotIn('Ram', df_new.columns)
        
        # The DataFrame should remain valid even if 'Screen' was missing
        self.assertIsInstance(df_new, pd.DataFrame)
        self.assertEqual(df_new.shape[1], 3) 

    # --- Edge Test Case [cite: 98] ---
    def test_empty_removal_list(self):
        """
        Tests the boundary condition where an empty list is provided.
        """
        columns_to_remove = []
        df_new = remove_columns(self.df, columns_to_remove)
        
        # The DataFrame should remain exactly the same
        pd.testing.assert_frame_equal(self.df, df_new)

    # --- Error Test Case [cite: 99] ---
    def test_invalid_input_type_columns(self):
        """
        Tests how the function handles invalid input types (e.g., string instead of list).
        The function is designed to catch TypeError and return the original DataFrame.
        """
        # Passing a string instead of a list
        result = remove_columns(self.df, "Not a list")
        
        # Expectation: The function catches the error and returns the original DF
        pd.testing.assert_frame_equal(self.df, result)

    def test_invalid_input_type_df(self):
        """
        Tests how the function handles invalid DataFrame input.
        """
        not_a_df = "I am a string"
        result = remove_columns(not_a_df, ['Price'])
        
        # Expectation: Returns the original input due to exception handling
        self.assertEqual(result, not_a_df)

    # --- Logical/Safety Test ---
    def test_typo_protection(self):
        """
        Verifies that a typo (e.g., 'price' instead of 'Price') does not cause
        accidental data loss. The function should suggest a fix but not delete.
        """
        columns_to_remove = ['price'] # Typo: Lowercase 'p'
        df_new = remove_columns(self.df, columns_to_remove)
        
        # The original 'Price' column must still exist
        self.assertIn('Price', df_new.columns)

if __name__ == '__main__':
    unittest.main()