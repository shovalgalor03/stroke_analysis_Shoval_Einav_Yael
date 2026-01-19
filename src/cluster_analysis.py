import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.patches as mpatches
from src.logger import setup_logger
import os

logger = setup_logger("cluster_analysis") # Initialize the logger
GLOBAL_PALETTE_NAME = "rocket" # ensures all plots use the same colors for the same groups

def prepare_data(df: pd.DataFrame) -> np.ndarray:
    """
    Internal helper to encode and scale data for clustering.
    """
    # 1. Validation
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # 2. Encoding and Scaling
    X_encoded = pd.get_dummies(df, drop_first=True)
    X_scaled = StandardScaler().fit_transform(X_encoded)
    
    return X_scaled

# --- Function 1: Find Optimal K (Elbow Method) ---
def find_optimal_k(X_scaled: np.ndarray, save_path: str, max_k: int = 10) -> int:
    """
    Plots the Elbow Method (Inertia) to determine the optimal number of clusters.
    Cleaned version for professional academic presentation.
    """
    logger.info(f"START: Searching for optimal K (range: 1 to {max_k}).")
    
    try:
        inertia = []
        silhouette_scores = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=1, n_init=10) # Initialize the model with k clusters
            clusters = kmeans.fit_predict(X_scaled) # assign a cluster label to each patient
            inertia.append(kmeans.inertia_) # sum of squared distances within the group
            
            if k >= 2:
                silhouette_scores.append(silhouette_score(X_scaled, clusters)) # Separation between the groups
            else:
                silhouette_scores.append(-1) # Placeholder for k=1, silhouette score requires at least 2 clusters
        
        # Determine best K    
        best_k_index = np.argmax(silhouette_scores)
        best_k = k_range[best_k_index]
        best_inertia = inertia[best_k_index]

        #  Plotting (Fixed Visualization)
        plt.figure(figsize=(10, 8)) 
        plt.plot(k_range, inertia, marker='o', linestyle='-', color="#2669de", linewidth=2, markersize=8, 
                 label='Inertia Trend') # Plot the Elbow line in blue
        plt.plot(best_k, best_inertia, marker='o', color='red', markersize=15, fillstyle='none', 
                 markeredgewidth=3, label=f'Optimal K: {best_k}') # Highlight the Best K in Red
        
        # Formatting for clarity
        plt.title('Elbow Method for Optimal K', fontsize=17, fontweight='bold')
        plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        plt.ylabel('Inertia (In-cluster sum of squares)', fontsize=12, fontweight='bold')
        plt.xticks(k_range) # Show all numbers from 1 to 10
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save with explicit .png extension
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info("Saved: elbow_method_k.png")

        logger.info(f"Optimization complete. Best K: {best_k}")
        return best_k

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 2 # Safe fallback
        
    except AssertionError as e:
        logger.error(f"Integrity Check Failed: {e}")
        raise e

    except Exception as e:
        logger.error(f"Optimization process failed: {e}")
        raise e

# --- Function 2: Perform Clustering ---
def perform_clustering(df: pd.DataFrame, X_scaled: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """
    Executes the full K-Means pipeline. 
    """
    logger.info(f"START: Starting K-Means pipeline with {n_clusters} clusters.")
    df_clustered = df.copy() # Work on a copy to ensure data integrity

    try:
        assert n_clusters >= 2, f"Input Error: n_clusters must be at least 2, got {n_clusters}."
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df_clustered['cluster'] = clusters # Saving Results

        df_clustered['group_name'] = df_clustered['cluster'].apply(lambda x: f"Group {x}") # Create generic group names (Group 0, Group 1, etc.)
         
        # Post-Validation
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
def plot_clusters_pca(X_scaled: np.ndarray, cluster_series: pd.Series, save_path):
    """
    Visualizes the clusters in 2D space using PCA.
    """
    logger.info("START: Generating PCA Visualization.")

    try:
        # 1. Apply PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_scaled)
        
        # 2. Log variance explanation for quality check
        exp_var = pca.explained_variance_ratio_
        logger.info(f"PC1 explains {exp_var[0]:.2%} of variance.")
        logger.info(f"PC2 explains {exp_var[1]:.2%} of variance.")
        logger.info(f"Total explained variance (2 components): {sum(exp_var):.2%}")
        
        # 3. Create DataFrame for plotting
        pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
        pca_df['Group'] = cluster_series.apply(lambda x: f"Group {x}")        
        
        n_colors = cluster_series.nunique()
        palette = sns.color_palette(GLOBAL_PALETTE_NAME, n_colors=n_colors)        
        
        hue_order = [f"Group {i}" for i in sorted(cluster_series.unique())]
        
        # 4. Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2',hue='Group', hue_order=hue_order, palette=palette, s=60, alpha=0.7)        
        
        plt.title(f'PCA Cluster Map (K={n_colors})', fontsize=14, fontweight='bold')
        plt.legend(title='Patient Group')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"PCA Visualization failed: {e}")

# --- Function 4
def calculate_cluster_risks(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Generic risk calculation per cluster.
    """
    df_clustered['stroke'] = pd.to_numeric(df_clustered['stroke'], errors='coerce')
    summary = df_clustered.groupby(['cluster', 'group_name'])['stroke'].mean().reset_index()
    summary.columns = ['cluster', 'group_name', 'stroke_risk_%']
    summary['stroke_risk_%'] *= 100
    return summary
        
# --- Function 5: Get Cluster Profiles Data ---
def get_cluster_profiles(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the clusters and prepares a formatted vertical DataFrame.
    Calculates N, numeric means, and categorical modes with percentages.
    """
    logger.info("Analyzing cluster characteristics...")
    try:
        n_counts = df_clustered['group_name'].value_counts()
        
        # 2. Numeric Analysis (Means)
        num_cols = ['stroke', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        existing_num = [c for c in num_cols if c in df_clustered.columns]
        numeric_profile = df_clustered.groupby('group_name')[existing_num].mean()

        # 3. Categorical Analysis (Mode + Percentage)
        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        existing_cat = [c for c in cat_cols if c in df_clustered.columns]
        
        cat_results = {}
        for group in df_clustered['group_name'].unique():
            group_data = df_clustered[df_clustered['group_name'] == group]
            group_stats = {}
            for col in existing_cat:
                mode_val = group_data[col].mode()[0]
                percentage = (group_data[col] == mode_val).mean() * 100
                group_stats[col] = f"{mode_val} ({percentage:.1f}%)"
            
            cat_results[group] = group_stats
        
        categorical_profile = pd.DataFrame(cat_results).T

        # 4. Assembly and Transpose to Vertical format
        combined = pd.concat([numeric_profile, categorical_profile], axis=1)
        combined.insert(0, 'N (Patients)', n_counts)
        
        return combined.T # Returns vertical table (Metrics as rows)
    
    except Exception as e:
        logger.error(f"Failed to plot styled table: {e}")
     

# --- Function 6: Plot Styled Profile Table (FIXED) ---
def plot_cluster_profile_table(profile_df: pd.DataFrame, save_path):
    """
    Renders a formatted vertical table as a high-quality PNG image.
    Fixed scaling and visibility issues.
    """
    logger.info(f"Generating styled table image: {save_path}")

    try:
        # 1. Adjust figure size based on the number of rows
        fig_height = max(6, len(profile_df) * 0.6)
        fig_width = max(8, len(profile_df.columns) * 2.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        # 2. Format numeric values for display
        display_df = profile_df.map(lambda x:
            f"{x:.2f}" if isinstance(x, (float, int)) else str(x))
        
        # 3. Create the table
        plt_table = ax.table(
            cellText=display_df.values,
            rowLabels=display_df.index,
            colLabels=display_df.columns,
            loc='center', cellLoc='center')

        # 4. Styling
        plt_table.auto_set_font_size(False)
        plt_table.set_fontsize(11)
        plt_table.scale(0.8, 2.2)
        
        header_colors = sns.color_palette(GLOBAL_PALETTE_NAME, n_colors=len(display_df.columns))
        for col_idx, color in enumerate(header_colors):
            cell = plt_table[0, col_idx]
            cell.set_facecolor(color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=12)

        for row_idx in range(len(display_df) + 1):
            try:
                # Row labels are at column -1
                label_cell = plt_table[row_idx, -1]
                label_cell.set_text_props(fontweight='bold', color='#2c3e50')
                label_cell.set_facecolor('#f2f2f2')
            except KeyError:
                continue

        plt.title('Clinical Profile Comparison', 
                  fontweight='bold', fontsize=16, pad=20, color='#2c3e50')
        
        # 5. Save and explicit SHOW for VS Code
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Table image saved successfully as: {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot styled table: {e}")
                        
# --- Function 7: Risk Bar Chart ---
def plot_risk_analysis(summary_table: pd.DataFrame, save_path):
    """
    Visualizes stroke risk per group using a tall and narrow Bar Chart.
    """
    logger.info("START: Generating Risk Bar Chart.")
    
    try:
        # Create mapping for group names
        n_groups = len(summary_table)
        palette = sns.color_palette(GLOBAL_PALETTE_NAME, n_colors=n_groups)        
        plt.figure(figsize=(max(5, n_groups * 2), 7))
        
        ax = sns.barplot(data=summary_table, 
                         x='group_name', y='stroke_risk_%', 
                         palette=palette, hue='group_name', legend=False)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')
        
        plt.title('Stroke Risk by Group', fontsize=14, fontweight='bold')
        plt.ylabel('Risk (%)')
        plt.xlabel('Patient Group')
        
        max_risk = summary_table['stroke_risk_%'].max()
        plt.ylim(0, max_risk * 1.2) 
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info("Saved: plot_risk_analysis.png") 

    except Exception as e:
        logger.error(f"Risk Chart failed: {e}")
        
# --- Function 8: Capture Rate Pie Chart (The "Catch") ---
def plot_stroke_capture_rate(df_clustered: pd.DataFrame, save_path):
    """
    Visualizes what percentage of total stroke patients fall into each group.
    """
    logger.info("START: Generating Capture Rate Pie Chart.")
    
    try:
        total_strokes = df_clustered['stroke'].sum()
        if total_strokes == 0:
            logger.warning("No stroke cases found. Skipping plot.")
            return

        capture_counts = df_clustered.groupby('group_name')['stroke'].sum().reset_index()
        capture_counts['capture_%'] = (capture_counts['stroke'] / total_strokes) * 100
        
        n_groups = capture_counts['group_name'].nunique()
        palette = sns.color_palette(GLOBAL_PALETTE_NAME, n_colors=n_groups)
        
        plt.figure(figsize=(9, 9))
        plt.pie(capture_counts['capture_%'], 
                labels=capture_counts['group_name'],
                autopct='%1.1f%%', 
                colors=palette,
                textprops={'fontsize': 18, 'weight': 'bold', 'color': 'white'},
                explode=[0.05] + [0]*(len(capture_counts)-1), 
                shadow=True, 
                startangle=90)
        
        # Add legend using patches to match bar chart colors
        legend_handles = [mpatches.Patch(color=palette[i], label=capture_counts['group_name'][i]) 
                        for i in range(n_groups)]
        plt.legend(handles=legend_handles, title="Patient Group", loc="upper left", bbox_to_anchor=(1, 1))            
        plt.title(f'Stroke Capture Rate\n(Total: {int(total_strokes)} patients)', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info("Saved: plot_stroke_capture_rate.png") 

    except Exception as e:
        logger.error(f"Capture Rate Plot failed: {e}")