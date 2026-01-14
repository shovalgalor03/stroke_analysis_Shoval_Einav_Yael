import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.logger import setup_logger

# Setup Logger
logger = setup_logger("visualization")

# ==========================================
# PART 1: CHI-SQUARE VISUALIZATIONS
# ==========================================

def plot_stacked_distribution(df, group_col='risk_group', target_col='stroke'):
    """
    1. Stacked Bar Chart: Shows the % of Stroke vs Healthy in each group.
    """
    logger.info("Generating Stacked Bar Chart")
    try:
        # Safety checks
        if df is None or df.empty:
            logger.warning("DataFrame is empty. Skipping Stacked Bar.")
            return

        # Create a contingency table converting counts to percentages per row (Risk Group)
        # normalize='index': Ensures each row sums to 100%
        ct = pd.crosstab(df[group_col], df[target_col], normalize='index') * 100

        # Plot a stacked bar chart using custom colors (Blue=Healthy, Red=Stroke)
        ax = ct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#8ecae6', '#d62828'])
        
        plt.title('Stroke Rate by Risk Group (Distribution)', fontsize=14)
        plt.ylabel('Percentage (%)')
        plt.xlabel('Risk Group')
        plt.xticks(rotation=45)

        # --- LEGEND POSITION ---
        # Move legend outside to the right to avoid overlapping data
        # bbox_to_anchor=(1.02, 1): Places it just outside the axes
        # loc='upper left': Anchors the legend's top-left corner to that point
        plt.legend(title='Outcome', labels=['Healthy', 'Stroke'], 
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        # Add percentage labels
        for container in ax.containers:
            # Format the label as a float with 1 decimal place followed by a '%' sign
            ax.bar_label(container, fmt='%.1f%%', label_type='center', color='black', fontweight='bold')

        # Adjust margins, save the image to disk, and free up memory
        plt.tight_layout()
        plt.savefig("1_stacked_bar.png")
        plt.close()
        logger.info("Saved: 1_stacked_bar.png")

    except Exception as e:
        logger.error(f"Stacked Bar Error: {e}")

def plot_mosaic_overview(df, group_col='risk_group', target_col='stroke'):
    """
    2. Mosaic Plot: Visualizes group sizes (width) AND stroke rates (height).
    Updated: Percentages INSIDE the boxes, Black borders, FORCED ROTATION.
    """
    logger.info("Generating Mosaic Plot...")

    try:
        if df is None or df.empty:
            logger.warning("DataFrame is empty. Skipping Mosaic Plot.")
            return
        
        # 1. Initialize Figure and Axes explicitly
        fig, ax = plt.subplots(figsize=(12, 7)) 
        
        # 2. Prepare Data
        cross_props = pd.crosstab(df[group_col], df[target_col], normalize='index') * 100
        label_map = {}
        for group in cross_props.index:
            if 0 in cross_props.columns:
                label_map[(str(group), '0')] = f"{cross_props.loc[group, 0]:.1f}%"
            if 1 in cross_props.columns:
                label_map[(str(group), '1')] = f"{cross_props.loc[group, 1]:.1f}%"

        def props(key):
            is_stroke = str(key[1]) == '1'
            return {
                'color': '#d62828' if is_stroke else '#8ecae6', 
                'alpha': 0.9, 'edgecolor': 'black', 'linewidth': 0.5}

        def labelizer(key):
            k = (str(key[0]), str(key[1]))
            return label_map.get(k, "") 
        
        # 3. Generate Plot (Pass 'ax' explicitly)
        mosaic(df, [group_col, target_col], properties=props, gap=0.007, 
               title='Mosaic Plot: Sample Size vs Outcome', labelizer=labelizer, ax=ax)
        
        for text in ax.texts:
            text.set_fontsize(16) 
        
        ax.set_title('Mosaic Plot: Sample Size vs Outcome', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Risk Group (Width = Sample Size)', fontsize=13)
        ax.set_ylabel('Outcome (Height = Proportion)', fontsize=13)
        
        # --- THE NUCLEAR FIX: Force Rotation using a Loop ---
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(12)      
            label.set_fontweight('bold')
        
        # Legend
        legend_elements = [Patch(facecolor='#8ecae6', edgecolor='black', label='Healthy (0)'),
                           Patch(facecolor='#d62828', edgecolor='black', label='Stroke (1)')]
        ax.legend(handles=legend_elements, loc='upper right', title="Outcome")
        
        plt.tight_layout()
        plt.savefig("2_mosaic_plot.png")
        plt.close()
        logger.info("Saved: 2_mosaic_plot.png")

    except Exception as e:
        logger.error(f"Mosaic Plot Error: {e}")

def plot_residuals_heatmap(df, group_col='risk_group', target_col='stroke'):
    """
    3. Pearson Residuals Heatmap
    """
    logger.info("Generating Residuals Heatmap...")
    try:
        if df is None or df.empty:
            logger.warning("DataFrame is empty. Skipping Heatmap.")
            return

        # Calculate Pearson Residuals
        obs = pd.crosstab(df[group_col], df[target_col])
        chi2, p, dof, expected = chi2_contingency(obs)
        residuals = (obs - expected) / np.sqrt(expected)

        plt.figure(figsize=(10, 6))
        
        # Black grid lines (linewidths=1, linecolor='black')
        sns.heatmap(residuals, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                    linewidths=1, linecolor='black')
        
        plt.title('Pearson Residuals Heatmap (Red = Higher than Expected)', fontsize=14)
        plt.ylabel('Risk Group')
        plt.xlabel('Outcome (0=Healthy, 1=Stroke)')
        
        plt.tight_layout()
        plt.savefig("3_residuals_heatmap.png")
        plt.close()
        logger.info("Saved: 3_residuals_heatmap.png")

    except Exception as e:
        logger.error(f"Heatmap Error: {e}")


# ==========================================
# PART 2: RELATIVE RISK (RR) VISUALIZATIONS
# ==========================================

def plot_rr_forest(results_df):
    """
    4. Forest Plot: Shows RR with Confidence Intervals.
    Updated: Text labels moved ABOVE the lines.
    """
    logger.info("Generating Forest Plot...")
    try:
        if results_df is None or results_df.empty:
            logger.warning("Results DataFrame is empty. Skipping Forest Plot.")
            return

        plt.figure(figsize=(10, 6))
        
        # Iterate through rows with index 'i' for plotting; check if CI excludes 1 (Significant)
        for i, row in results_df.iterrows():
            is_significant = (row['CI_Lower'] > 1) or (row['CI_Upper'] < 1)
            color = '#d62828' if is_significant else 'black'
            
            # Plot whiskers defined by xerr; 'ecolor' sets the color of the error bars lines
            plt.errorbar(x=row['RR'], y=row['comparison'], 
                         xerr=[[row['RR']-row['CI_Lower']], [row['CI_Upper']-row['RR']]], 
                         fmt='o', color=color, ecolor=color, capsize=5)
            
            # Text Label (Moved UP: i + 0.15 to avoid overlapping the line)
            label_text = f"RR={row['RR']:.2f}\n({row['CI_Lower']:.2f}-{row['CI_Upper']:.2f})"
            plt.text(row['RR'], i + 0.15, label_text, va='bottom', ha='center', 
                     fontsize=9, fontweight='bold', color='black')

        plt.axvline(x=1, color='black', linestyle='--', label='No Effect (RR=1)')
        
        # Adjust Y limits to make room for the top label
        plt.ylim(-0.5, len(results_df) - 0.5 + 0.5)
        
        # Custom Legend
        custom_lines = [Line2D([0], [0], color='#d62828', marker='o', linestyle=''),
                        Line2D([0], [0], color='black', marker='o', linestyle='')]
        plt.legend(custom_lines, ['Significant', 'Not Significant'], loc='lower right')

        plt.title('Relative Risk Forest Plot (Red=Significant)', fontsize=14)
        plt.xlabel('Relative Risk (RR)')
        plt.yticks(range(len(results_df)), results_df['comparison'])
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig("4_forest_plot.png")
        plt.close()
        logger.info("Saved: 4_forest_plot.png")

    except Exception as e:
        logger.error(f"Forest Plot Error: {e}")


def plot_rr_lollipop(results_df):
    """
    5. Lollipop Plot: Compare Effect Sizes.
    Updated: Red Circle -> Black Text. Black Circle -> White Text.
    """
    logger.info("Generating Lollipop Plot...")
    try:
        if results_df is None or results_df.empty:
            logger.warning("Results DataFrame is empty. Skipping Lollipop Plot.")
            return
        
        data = results_df.sort_values('RR', ascending=True)

        plt.figure(figsize=(10, 6))
        
        # Enumerate gives us the index 'i' for Y-axis positioning
        for i, (idx, row) in enumerate(data.iterrows()):
            is_significant = (row['CI_Lower'] > 1) or (row['CI_Upper'] < 1)
            
            # Color Logic: High contrast for text inside markers
            if is_significant:
                marker_color = '#d62828' # Red Marker
                text_color = 'black'     # Black Text inside Red
            else:
                marker_color = 'black'   # Black Marker
                text_color = 'white'     # White Text inside Black
            
            # Draw Stem
            plt.hlines(y=row['comparison'], xmin=1, xmax=row['RR'], color='skyblue', linewidth=3)
            
            # Draw Head (Big Marker)
            plt.plot(row['RR'], row['comparison'], "o", markersize=28, color=marker_color, alpha=1.0)
            
            # Draw Text Inside Marker
            plt.text(row['RR'], row['comparison'], f"{row['RR']:.2f}", 
                     va='center', ha='center', fontweight='bold', color=text_color, fontsize=9)

        plt.axvline(x=1, color='black', linestyle='--')
        plt.title('Relative Risk Magnitude (Lollipop Chart)', fontsize=14)
        plt.xlabel('Relative Risk (RR)')
        
        plt.tight_layout()
        plt.savefig("5_lollipop_plot.png")
        plt.close()
        logger.info("Saved: 5_lollipop_plot.png")

    except Exception as e:
        logger.error(f"Lollipop Plot Error: {e}")


# ==========================================
# MASTER FUNCTION
# ==========================================

def plot_all_visualizations(df, results_df):
    """
    Orchestrator: Runs all 5 visualizations sequentially.
    """
    logger.info("--- Starting Visualization Pipeline ---")
    
    plot_stacked_distribution(df)
    plot_mosaic_overview(df)
    plot_residuals_heatmap(df)
    
    if results_df is not None and not results_df.empty:
        plot_rr_forest(results_df)
        plot_rr_lollipop(results_df)
    else:
        logger.warning("No results_df provided. Skipping RR plots.")
        
    logger.info("--- Visualization Pipeline Completed ---")