import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.logger import setup_logger

logger = setup_logger("Cluster_Analysis")

def prepare_data_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Encode categorical data and scale features."""
    try:
        # Drop target and ID columns
        cols_drop = [c for c in ['stroke', 'id'] if c in df.columns]
        X = df.drop(columns=cols_drop)
        
        # Convert text to numbers (Encoding)
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Scale data (Critical for K-Means!)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
        
        return X_scaled
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise e

def run_kmeans(df_original: pd.DataFrame, X_scaled: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Step 2: Run K-Means and assign clusters to original data."""
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clustered = df_original.copy()
        df_clustered['cluster'] = clusters
        
        logger.info(f"Successfully divided data into {n_clusters} clusters.")
        return df_clustered
    except Exception as e:
        logger.error(f"Error in K-Means execution: {e}")
        raise e

def analyze_risk_factors(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Analyze stroke risk per cluster (The 'Punch')."""
    try:
        # Group by cluster and calculate stats
        summary = df_clustered.groupby('cluster').agg({
            'stroke': lambda x: x.mean() * 100,  # Risk %
            'age': 'mean',
            'bmi': 'mean',
            'avg_glucose_level': 'mean'
        }).reset_index()
        
        summary = summary.rename(columns={'stroke': 'risk_percent'})
        return summary.sort_values('risk_percent', ascending=False)
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise e