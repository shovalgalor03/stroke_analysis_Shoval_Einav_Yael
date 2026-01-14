import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.logger import setup_logger

logger = setup_logger("cluster_analysis") # Initialize the logger

# --- Function 1: Find Optimal K (Elbow Method) ---
def find_optimal_k(df: pd.DataFrame, max_k: int = 10) -> int:
    """
        Plots the Elbow Method (Inertia) to help determine the optimal number of clusters.
        Returns the statistically best K (calculated internally using Silhouette, 
        but only showing Elbow plot as requested).
        """
    logger.info(f"START: Searching for optimal K (range: 0 to {max_k}).")

    try:
        # 1. Validation checks
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        assert max_k >= 2, f"Input Error: max_k must be at least 2, got {max_k}."
        assert len(df) > max_k, "Data Error: Not enough data points for the requested number of clusters."

        # 2. Data Preparation (Encoding + Scaling)
        logger.info("Preparing data for optimization analysis...")
        X_encoded = pd.get_dummies(df, drop_first=True)
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_encoded), columns=X_encoded.columns)

        # 3. Calculation Loop
        inertia = []
        silhouette_scores = []
        k_range = range(1, max_k + 1)

        logger.info("Calculating metrics for each K...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            
            if k >= 2:
                silhouette_scores.append(silhouette_score(X_scaled, labels))
            else:
            # Placeholder for k=1 since silhouette is not defined
                silhouette_scores.append(-1)
            
        # 4. Plotting (Only Elbow Method)
        logger.info("Generating Elbow plot...")
        plt.figure(figsize=(14, 8))
        plt.plot(k_range, inertia, marker='o', linestyle='--', color='b', linewidth=2)
        
        padding_y = (max(inertia) - min(inertia)) * 0.2  # Adjust Y-axis: Focus on the actual data range rather than starting from 0
        plt.ylim(min(inertia) - padding_y, max(inertia) + padding_y)
        
        plt.xlim(min(k_range) - 0.5, max(k_range) + 0.5) # Adjust X-axis: Set limits strictly to the range of k
        plt.xticks(k_range)
        
        plt.title('Elbow Method (Inertia Analysis)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Lower is better)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("Elbow Method")
        plt.close()
        logger.info("Saved: Elbow Method")

        # 5. Determine best K automatically
        # We use the silhouette max internally to give a good default suggestion
        best_k_index = np.argmax(silhouette_scores)
        best_k = k_range[best_k_index]
        
        logger.info(f"Optimization complete. Calculated Best K: {best_k}")
        return best_k

    except Exception as e:
        logger.error(f"Optimization process failed: {e}")
        return 2 # Fallback default
        
    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        raise e

    except Exception as e:
        logger.error(f"Optimization process failed: {e}")
        raise e

# --- Function 2: Perform Clustering ---
def perform_clustering(df: pd.DataFrame, n_clusters: int = 2) -> pd.DataFrame:
    """
    Executes the full K-Means pipeline. 
    """
    logger.info(f"START: Starting K-Means pipeline with {n_clusters} clusters.")
    df_clustered = df.copy() # Work on a copy to ensure data integrity

    try:
        # 1. Validation
        if df_clustered.empty:
            raise ValueError("Input DataFrame is empty.")
        
        assert n_clusters >= 2, f"Input Error: n_clusters must be at least 2, got {n_clusters}."

        # 2. Encoding & Scaling
        X_encoded = pd.get_dummies(df_clustered, drop_first=True)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
        
        # 3. Running Model
        kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 4. Saving Results
        df_clustered['cluster'] = clusters

        # 5. Post-Validation
        assert 'cluster' in df_clustered.columns, "Verification Failed: 'cluster' column was not created."
        assert not df_clustered['cluster'].isna().any(), "Verification Failed: Null values found in 'cluster' column."

        logger.info("SUCCESS: Clustering pipeline complete.")
        return df_clustered

    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        raise e

    except Exception as e:
        logger.error(f"Clustering pipeline failed: {e}")
        raise e

# --- Function 3: Visualize with PCA (The "Map") ---
def plot_clusters_pca(df_clustered: pd.DataFrame):
    """
    Visualizes the clusters in 2D space using PCA.
    """
    logger.info("START: Generating PCA Visualization.")

    try:
        # 1. Prepare data (Drop cluster column for PCA calculation)
        feature_cols = df_clustered.drop(columns=['cluster'])
        X_encoded = pd.get_dummies(feature_cols, drop_first=True)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # 2. Apply PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_scaled)

        # 3. Create DataFrame for plotting
        pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
        pca_df['cluster'] = df_clustered['cluster'].values

        # 4. Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', 
            hue='cluster', data=pca_df, 
            palette='viridis', s=60, alpha=0.7)
        
        plt.title('Patient Segments Visualization (PCA)', fontsize=14)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster Group')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plot_clusters_pca")
        plt.close()
        logger.info("Saved: plot_clusters_pca")

    except Exception as e:
        logger.error(f"PCA Visualization failed: {e}")

# --- Function 4: Profile Analysis ---
def analyze_cluster_profile(df_clustered: pd.DataFrame):
    """
    Generates a statistical profile for each cluster.
    - Numeric columns: Calculates Mean.
    - Categorical columns: Calculates Mode (Most Common Value).
    """
    logger.info("START: Generating Cluster Profile Report...")

    try:
        # --- Part A: Numeric Features (Calculate Average) ---
        numeric_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']
        
        existing_numeric = [col for col in numeric_cols if col in df_clustered.columns] # Filter to keep only columns that actually exist
        
        if existing_numeric:
            logger.info("Calculating numeric averages (Profile):")
            numeric_profile = df_clustered.groupby('cluster')[existing_numeric].mean().round(2) # Calculate mean per cluster
            numeric_profile['Count'] = df_clustered['cluster'].value_counts() # Add patient count
            
            logger.info("\n" + numeric_profile.to_string()) # Log the table

        # --- Part B: Categorical Features (Calculate Most Common Value) ---
        categorical_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status', 'ever_married']
        
        existing_categorical = [col for col in categorical_cols if col in df_clustered.columns] # Filter to keep only columns that actually exist

        if existing_categorical:
            logger.info("Calculating dominant categorical features (Mode):")
            
            def get_mode(x): # Helper function to find the 'Mode' (most frequent value)
                return x.mode()[0] if not x.mode().empty else 'N/A'
            
            categorical_profile = df_clustered.groupby('cluster')[existing_categorical].agg(get_mode) # Aggregate using the mode function
            
            logger.info("\n" + categorical_profile.to_string()) # Log the table

    except Exception as e:
        logger.error(f"Profiling failed: {e}")

# --- Function 5: Risk Bar Chart ---
def plot_risk_analysis(summary_table: pd.DataFrame):
    """
    Visualizes stroke risk per cluster using a Bar Chart.
    """
    logger.info("START: Generating Risk Bar Chart.")

    try:
        required_cols = ['cluster', 'stroke_risk_%']
        for col in required_cols:
            if col not in summary_table.columns:
                raise KeyError(f"Missing required column: '{col}'")

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=summary_table, 
            x='cluster', y='stroke_risk_%', palette='Reds')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')
        
        plt.title('Stroke Risk by Cluster', fontsize=14, fontweight='bold')
        plt.ylabel('Risk (%)')
        plt.xlabel('Cluster Group')
        
        # Dynamic Y-limit
        max_risk = summary_table['stroke_risk_%'].max()
        plt.ylim(0, max_risk * 1.2) 
        
        plt.tight_layout()
        plt.savefig("plot_risk_analysis")
        plt.close()
        logger.info("Saved: plot_risk_analysis") 

    except Exception as e:
        logger.error(f"Risk Chart failed: {e}")

# --- Function 6: Capture Rate Pie Chart (The "Catch") ---
def plot_stroke_capture_rate(df_clustered: pd.DataFrame):
    """
    Visualizes the 'Capture Rate': What percentage of TOTAL stroke patients 
    were found in each cluster?
    """
    logger.info("START: Generating Capture Rate Pie Chart.")
    
    try:
        # 1. Total strokes in dataset
        total_strokes = df_clustered['stroke'].sum()
        
        if total_strokes == 0:
            logger.warning("No stroke cases found. Cannot plot capture rate.")
            return

        # 2. Strokes per cluster
        capture_counts = df_clustered.groupby('cluster')['stroke'].sum().reset_index()
        
        # 3. Calculate percentage
        capture_counts['capture_%'] = (capture_counts['stroke'] / total_strokes) * 100
        
        # 4. Plotting (Pie Chart)
        plt.figure(figsize=(7, 7))
        
        # Sort so the largest slice (highest risk) comes first
        capture_counts = capture_counts.sort_values('capture_%', ascending=False)
        colors = ['#ff6666', '#66b3ff', '#99ff99'] # Red, Blue, Green
        
        plt.pie(
            capture_counts['capture_%'], 
            labels=capture_counts['cluster'].astype(str) + '\n(' + capture_counts['capture_%'].round(1).astype(str) + '%)',
            colors=colors,
            explode=[0.05] + [0]*(len(capture_counts)-1), # Highlight the biggest slice
            shadow=True,
            startangle=90
        )
        
        plt.title(f'Stroke Capture Rate\n(Coverage of {int(total_strokes)} total patients)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("plot_stroke_capture_rate")
        plt.close()
        logger.info("Saved: plot_stroke_capture_rate") 


    except Exception as e:
        logger.error(f"Capture Rate Plot failed: {e}")


