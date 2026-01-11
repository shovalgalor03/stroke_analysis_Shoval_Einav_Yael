import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.logger import setup_logger

logger = setup_logger("cluster_analysis")

def run_cluster_analysis(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Performs K-Means clustering on numeric features and visualizes results using PCA.
    Useful for identifying latent patient groups (risk profiling).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        n_clusters (int): The number of clusters to generate (default is 3).

    Returns:
        pd.DataFrame: A copy of the DataFrame with a new 'Cluster' column.
                      Returns original DataFrame if the process fails.
    """
    logger.info(f"START: Running Cluster Analysis with {n_clusters} clusters.")

    try:
        # 1. Input Validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        df_analyzed = df.copy()

        # 2. Feature Selection
        # We drop ID and Target (stroke) to find natural patterns in the health data itself.
        cols_to_exclude = ['id', 'stroke']
        
        # Select only numeric columns for K-Means (it cannot handle strings directly)
        features = df_analyzed.drop(columns=cols_to_exclude, errors='ignore')
        features = features.select_dtypes(include=[np.number])

        if features.empty:
            logger.warning("No numeric features found for clustering. Aborting analysis.")
            return df

        # Check for NaNs (K-Means does not support missing values)
        if features.isna().sum().sum() > 0:
            logger.error("Data contains missing values (NaN). Please run 'fill_missing' first.")
            return df

        # 3. Scaling (Critical for K-Means)
        logger.info(f"Scaling data using StandardScaler on {features.shape[1]} features...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # 4. Run K-Means
        logger.info("Executing K-Means algorithm...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)

        # Add result back to DataFrame
        df_analyzed['Cluster'] = clusters
        logger.info("Successfully assigned clusters to patients.")

        # 5. Insight Generation: Stroke Rate per Cluster
        if 'stroke' in df_analyzed.columns:
            stroke_rates = df_analyzed.groupby('Cluster')['stroke'].mean()
            logger.info(f"Stroke rates per cluster:\n{stroke_rates}")

        # 6. Visualization with PCA (Dimensionality Reduction)
        try:
            logger.info("Generating PCA visualization (2D projection)...")
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_features)
            
            # Prepare plotting data
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x='PC1', y='PC2', 
                hue='Cluster', 
                data=pca_df, 
                palette='viridis', 
                s=100, alpha=0.7
            )
            plt.title(f'Patient Clusters (PCA) - {n_clusters} Groups')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(title='Cluster')
            plt.show() # Note: This will open a window and pause execution until closed
            logger.info("Visualization displayed successfully.")

        except Exception as plot_err:
            logger.warning(f"Visualization failed (analysis continues): {plot_err}")

        return df_analyzed

    except ValueError as e:
        logger.error(f"Value Error in clustering: {e}")
        return df

    except TypeError as e:
        logger.error(f"Type Error: {e}")
        return df

    except Exception as e:
        logger.exception(f"Critical error during cluster analysis: {e}")
        return df