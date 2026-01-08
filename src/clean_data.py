def drop_irrelevant_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame if they exist.
    Useful for cleaning non-predictive features (e.g., IDs) before analysis.
    """
    # Create a copy to avoid SettingWithCopyWarning on original dataframe
    df_clean = df.copy()

    if not columns_to_remove:
            logger.warning("No columns provided to remove (empty list).")
            return df_clean
    try:
        # --- Identify which columns actually exist ---
        existing_cols = []
        missing_cols = []  
        
        # Iterate over the requested columns and check if they exist in the dataset
        for col in columns_to_remove:
            if col in df_clean.columns:
                existing_cols.append(col)
            else:
                missing_cols.append(col)
                
        if missing_cols:
            logger.warning(f"Note: The following columns were not found and could not be dropped: {missing_cols}")
            
        if existing_cols:
            # Perform the drop operation
            df_clean = df_clean.drop(columns=existing_cols)
            logger.info(f"Successfully dropped columns: {existing_cols}")

        return df_clean

    except Exception as e:
        # Log the specific error and re-raise it to stop execution if critical
        logger.error(f"Critical error in drop_irrelevant_columns: {e}")
        raise e