import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from src.logger import setup_logger

logger = setup_logger("Cluster_Analysis")

def perform_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Executes the full K-Means pipeline:
    1. Prepares data (Encoding + Scaling).
    2. Runs K-Means algorithm.
    3. Returns original DataFrame with a new 'cluster' column.
    
    Assumption: Input 'df' contains only relevant features (ID/Target removed in main with 'remove_columns' function).
    """
    try:
        logger.info(f"Starting K-Means pipeline with {n_clusters} clusters...")

        #1A. Encoding - Convert text columns to numbers (Gender -> Gender_Male)
        X_encoded = pd.get_dummies(df, drop_first=True)     # drop_first=True prevents multicollinearity
        
        #1B. Scaling - Normalize features 
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
        
        #2. K-Means 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)         # n_init=10: Runs algorithm 10 times to find best centroids
        clusters = kmeans.fit_predict(X_scaled)
        
        #3. Combine Results 
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        logger.info("Clustering pipeline complete.")
        return df_clustered

    except Exception as e:
        logger.error(f"Clustering pipeline failed: {e}")
        raise e

def plot_clusters_pca(df_clustered: pd.DataFrame):
    """
    Visualizes the clusters using PCA (2D projection).
    """
    try:
        if 'cluster' not in df_clustered.columns:
            raise KeyError("Missing 'cluster' column. Run K-Means first.")

        logger.info("Generating PCA plot...")

        # 1. Prepare data for PCA (Compact version)
        # We need numerical data, so we re-encode and scale strictly for plotting
        X = df_clustered.drop(columns=['cluster'])
        X_encoded = pd.get_dummies(X, drop_first=True)
        X_scaled = StandardScaler().fit_transform(X_encoded) # Chained for brevity

        # 2. Run PCA to get 2D coordinates
        pca_coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

        # 3. Plotting
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=pca_coords[:, 0], 
            y=pca_coords[:, 1],
            hue=df_clustered['cluster'].astype(str), # Convert to string for discrete colors
            palette='viridis', 
            s=80, alpha=0.7
        )
        
        plt.title('Cluster Visualization (PCA Projection)')
        plt.xlabel('PC1'); plt.ylabel('PC2') # Semicolon allows two commands in one line
        plt.grid(True, alpha=0.3)
        plt.legend(title='Cluster')
        plt.show()

    except Exception as e:
        logger.error(f"Visualization failed: {e}")

def plot_risk_analysis(summary_table: pd.DataFrame):
    """
    Visualizes stroke risk per cluster using a Bar Chart.
    """
    try:
        logger.info("Generating Risk Bar Chart...")
        
        plt.figure(figsize=(8, 6))
        
        # 1. Create Bar Plot
        # We capture the axes object ('ax') to add labels later
        ax = sns.barplot(
            data=summary_table, 
            x='cluster', 
            y='stroke_risk_%', 
            palette='Reds'
        )
        
        # 2. Add Labels on top of bars
        # Uses the modern matplotlib API to auto-label bars
        ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3, fontweight='bold')

        # 3. Styling
        plt.title('Stroke Risk by Cluster', fontsize=14, fontweight='bold')
        plt.ylabel('Risk (%)')
        plt.xlabel('Cluster Group')
        
        # Add some breathing room at the top for the labels
        plt.ylim(0, summary_table['stroke_risk_%'].max() * 1.2) 
        
        plt.show()

    except Exception as e:
        logger.error(f"Risk Chart failed: {e}")






