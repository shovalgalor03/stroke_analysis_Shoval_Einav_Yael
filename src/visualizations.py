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
    Updated: X-axis labels now include the sample size (n).
    """
    logger.info("Generating Mosaic Plot...")

    try:
        if df is None or df.empty:
            logger.warning("DataFrame is empty. Skipping Mosaic Plot.")
            return

        # --- 1. Prepare Labels with Counts (NEW) ---
        # Calculate how many people are in each group
        counts = df[group_col].value_counts()

        # Helper function to add (n=...) to the label
        def make_label(k):
            if k in counts:
                return f"{k}\n(n={counts[k]})"
            return k

        # Create a copy and update the column with the new labels
        plot_df = df.copy()
        plot_df[group_col] = plot_df[group_col].apply(make_label)
        
        # --- 2. Initialize Figure and Axes explicitly ---
        fig, ax = plt.subplots(figsize=(12, 7)) 
        
        # --- 3. Prepare Data (Using plot_df to match new labels) ---
        cross_props = pd.crosstab(plot_df[group_col], plot_df[target_col], normalize='index') * 100
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
        
        # --- 4. Generate Plot (Pass 'ax' explicitly) ---
        # Note: Using plot_df here!
        mosaic(plot_df, [group_col, target_col], properties=props, gap=0.007, 
               title='Mosaic Plot: Sample Size vs Outcome', labelizer=labelizer, ax=ax)
        
        for text in ax.texts:
            text.set_fontsize(16) 
        
        ax.set_title('Mosaic Plot: Sample Size vs Outcome', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Risk Group (Width = Sample Size)', fontsize=13)
        ax.set_ylabel('Outcome (Height = Proportion)', fontsize=13)
        
        # --- Force Formatting using a Loop ---
        for label in ax.get_xticklabels():
            label.set_rotation(0) # Changed to 0 (horizontal) for better readability
            label.set_horizontalalignment('center')
            label.set_fontsize(11)      
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

from matplotlib.lines import Line2D

from matplotlib.lines import Line2D

from matplotlib.lines import Line2D

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

# ==========================================================
# PART 3: SCENARIO SPECIFIC VISUALIZATION WITHOUT OUTLIERS
# ==========================================================


def plot_results_table(results_df, title):
    """
    Displays the results DataFrame as a static image table.
    This allows the table to appear in the 'Plots' pane of IDEs.
    """
    logger.info(f"Generating Results Table for: {title}")
    
    if results_df.empty:
        return

    # 1. Select and Rename Columns for cleaner display
    display_df = results_df[['comparison', 'RR', 'P_Value', 'Significant_0.05', 'Outliers_Removed']].copy()
    display_df.columns = ['Group', 'RR', 'P Value', 'Significant_0.05', 'Outliers Removed']
    
    # Round numbers for better visuals
    display_df['RR'] = display_df['RR'].round(2)
    display_df['P Value'] = display_df['P Value'].round(4)

    # 2. Create Figure
    fig, ax = plt.subplots(figsize=(12, 3)) # Short height for a table
    ax.axis('off') # Turn off the X/Y axis lines

    # 3. Create the Table
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1] # Stretch to fill frame
    )

    # 4. Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Add Colors to Headers
    for (i, j), cell in table.get_celld().items():
        if i == 0: # Header Row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # Dark Blue background
        else:
            cell.set_facecolor('#f5f5f5') # Light Gray background for data

    # 5. Add Title
    plt.title(f"Results Table: {title}", fontsize=12, fontweight='bold', pad=10)
    
    # 6. Save and Show
    plt.tight_layout()
    filename = f"table_{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=200)
    logger.info(f"Saved: {filename}")


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
        plot_rr_lollipop(results_df)
    else:
        logger.warning("No results_df provided. Skipping RR plots.")
        
    logger.info("--- Visualization Pipeline Completed ---")
    