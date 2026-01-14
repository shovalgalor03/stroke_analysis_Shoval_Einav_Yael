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
            
        # 4. Plotting (Elbow Method)
        logger.info("Generating Elbow plot...")
        # Visual Adjustment 1: Tall and narrow aspect ratio 
        # (Stretches the graph vertically to emphasize the slope)
        plt.figure(figsize=(6, 9))
        
        # Visual Adjustment 2: Thicker lines and larger markers
        plt.plot(k_range, inertia, marker='o', linestyle='-', color='b', 
                 linewidth=4, markersize=12, markeredgecolor='black')
        
        # Visual Adjustment 3: Aggressive Zoom on X-Axis
        # We cut off the "long tail" (showing only up to 4.5). 
        # This removes the flattening effect at higher k values.
        plt.xlim(0.8, 4.5) 
        
        # Visual Adjustment 4: Crop the "Dead Space" on Y-Axis
        # Instead of starting from 0, we start just below the minimum value.
        # This forces the drop to occupy the entire vertical space of the plot.
        min_val = min(inertia)
        max_val = max(inertia)
        plt.ylim(min_val * 0.90, max_val * 1.05)
        
        # Formatting
        plt.title('Elbow Method (Inertia Analysis)', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        plt.ylabel('Inertia (Lower is better)', fontsize=12, fontweight='bold')
        
        # horizontal grid only (to help compare heights)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Show only relevant integer ticks on X (1, 2, 3, 4)
        plt.xticks(range(1, 5))
        
        plt.tight_layout()
        plt.savefig("Elbow_Method.png")
        plt.close()
        logger.info("Saved: Elbow_Method.png")

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
        
# --- Function 4: Profile Analysis (Auto-fit & Bold Design) ---
def analyze_cluster_profile(df_clustered: pd.DataFrame):
    """
    Generates a professionally styled vertical profile table with bold headers 
    and auto-fitted column widths.
    """
    logger.info("START: Generating Bold & Auto-fitted Vertical Profile Table.")

    try:
        # 1. Data Prep and Mapping
        df_p = df_clustered.copy()
        df_p['group'] = df_p['cluster'].map({0: 'Group A', 1: 'Group B'})
        n_counts = df_p['group'].value_counts()
        
        # 2. Numeric Analysis (Prioritizing N and Stroke Risk)
        num_cols = ['stroke', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        existing_num = [c for c in num_cols if c in df_p.columns]
        numeric_profile = df_p.groupby('group')[existing_num].mean()

        # 3. Categorical Analysis (Showing % to prove uniqueness)
        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        existing_cat = [c for c in cat_cols if c in df_p.columns]
        
        cat_results = {}
        for group in ['Group A', 'Group B']:
            group_data = df_p[df_p['group'] == group]
            group_stats = {}
            for col in existing_cat:
                mode_val = group_data[col].mode()[0]
                perc = (group_data[col] == mode_val).mean() * 100
                group_stats[col] = f"{mode_val} ({perc:.1f}%)"
            cat_results[group] = group_stats
        
        categorical_profile = pd.DataFrame(cat_results).T

        # 4. Final Data Assembly
        combined = pd.concat([numeric_profile, categorical_profile], axis=1)
        combined.insert(0, 'N (Patients)', n_counts)
        vertical_table = combined.T
        vertical_table.columns = ['Group A', 'Group B']

        # 5. Visual Rendering with Matplotlib
        fig, ax = plt.subplots(figsize=(12, 14)) # Slightly wider for auto-fit
        ax.axis('off')

        # Formatter for numbers
        display_df = vertical_table.map(
            lambda x: f"{x:.2f}" if isinstance(x, (float, int)) and not isinstance(x, bool) else str(x)
        )

        # Create table
        plt_table = ax.table(
            cellText=display_df.values,
            rowLabels=display_df.index,
            colLabels=display_df.columns,
            loc='center',
            cellLoc='center'
        )

        # --- ADVANCED STYLING: BOLD & AUTO-FIT ---
        plt_table.auto_set_font_size(False)
        plt_table.set_fontsize(12)
        
        # Auto-adjust column widths based on text length
        plt_table.auto_set_column_width(col=list(range(-1, len(display_df.columns)))) 
        
        # Vertical scaling for readability
        plt_table.scale(1.2, 3.5)

        # Style Headers (Group A, Group B)
        header_colors = ['#fc9272', '#de2d26']
        for col_idx, color in enumerate(header_colors):
            cell = plt_table[0, col_idx]
            cell.set_facecolor(color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=14)

        # Style Row Labels (The Index column)
        for row_idx in range(len(display_df) + 1):
            cell = plt_table[row_idx, -1] # Index cell
            cell.set_text_props(fontweight='bold', color='#2c3e50')
            cell.set_facecolor('#f2f2f2')

        # Zebra Stripes for rows
        for row_idx in range(1, len(display_df) + 1):
            if row_idx % 2 == 0:
                for col_idx in range(len(display_df.columns)):
                    plt_table[row_idx, col_idx].set_facecolor('#fafafa')

        plt.title('Clinical Profile Comparison\nGroup A vs Group B', 
                  fontweight='bold', fontsize=18, pad=60, color='#2c3e50')
        
        plt.savefig("cluster_profile_table.png", bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved Final Styled Table: cluster_profile_table.png")

    except Exception as e:
        logger.error(f"Styled profiling failed: {e}")
        
# --- Function 5: Risk Bar Chart ---
def plot_risk_analysis(summary_table: pd.DataFrame):
    """
    Visualizes stroke risk per group using a tall and narrow Bar Chart.
    """
    logger.info("START: Generating Risk Bar Chart.")
    GROUP_COLORS = {'Group A': '#fc9272', 'Group B': '#de2d26'}
    
    try:
        # Create mapping for group names
        plot_data = summary_table.copy()
        plot_data['group_name'] = plot_data['cluster'].map({0: 'Group A', 1: 'Group B'})

        # Dimensions: Narrow (5) and Taller (7)
        plt.figure(figsize=(5, 7))
        
        ax = sns.barplot(data=plot_data, 
                         x='group_name', y='stroke_risk_%', 
                         palette=GROUP_COLORS, hue='group_name', legend=False)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')
        
        plt.title('Stroke Risk by Group', fontsize=14, fontweight='bold')
        plt.ylabel('Risk (%)')
        plt.xlabel('Patient Group')
        
        max_risk = plot_data['stroke_risk_%'].max()
        plt.ylim(0, max_risk * 1.2) 
        
        plt.tight_layout()
        plt.savefig("plot_risk_analysis.png")
        plt.close()
        logger.info("Saved: plot_risk_analysis.png") 

    except Exception as e:
        logger.error(f"Risk Chart failed: {e}")
        
# --- Function 6: Capture Rate Pie Chart (The "Catch") ---
def plot_stroke_capture_rate(df_clustered: pd.DataFrame):
    """
    Visualizes what percentage of total stroke patients fall into each group.
    """
    logger.info("START: Generating Capture Rate Pie Chart.")
    GROUP_COLORS = {'Group A': '#fc9272', 'Group B': '#de2d26'}
    
    try:
        total_strokes = df_clustered['stroke'].sum()
        if total_strokes == 0:
            logger.warning("No stroke cases found. Skipping plot.")
            return

        capture_counts = df_clustered.groupby('cluster')['stroke'].sum().reset_index()
        capture_counts['capture_%'] = (capture_counts['stroke'] / total_strokes) * 100
        capture_counts['group_name'] = capture_counts['cluster'].map({0: 'Group A', 1: 'Group B'})
        
        capture_counts = capture_counts.sort_values('capture_%', ascending=False)
        colors = [GROUP_COLORS[g] for g in capture_counts['group_name']]

        plt.figure(figsize=(8, 8))
        
        plt.pie(capture_counts['capture_%'], 
                labels=capture_counts['group_name'],
                autopct='%1.1f%%', 
                colors=colors,
                textprops={'fontsize': 18, 'weight': 'bold', 'color': 'white'},
                explode=[0.05] + [0]*(len(capture_counts)-1), 
                shadow=True, 
                startangle=90)
        
        # Add legend using patches to match bar chart colors
        legend_handles = [mpatches.Patch(color=color, label=group) for group, color in GROUP_COLORS.items()]
        plt.legend(handles=legend_handles, title="Patient Group", loc="upper right", bbox_to_anchor=(1.15, 1))
              
        plt.title(f'Stroke Capture Rate\n(Total: {int(total_strokes)} patients)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("plot_stroke_capture_rate.png")
        plt.close()
        logger.info("Saved: plot_stroke_capture_rate.png") 

    except Exception as e:
        logger.error(f"Capture Rate Plot failed: {e}")