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
# PART 3: SCENARIO SPECIFIC VISUALIZATIONS WITHOUT OUTLIERS
# ==========================================================

def plot_rr_forest(results_df):
    """
    4. Forest Plot: Shows RR with Confidence Intervals.
    FORCE ORDER: BMI (Bottom) -> Glucose -> Both High (Top)
    """
    logger.info("Generating Forest Plot...")
    
    if results_df is None or results_df.empty:
        logger.warning("Results DataFrame is empty. Skipping Forest Plot.")
        return

    # --- Step 1: Force Order by Manual Reconstruction ---
    # Identify the column name (checks if it is 'Risk Factor' or 'comparison')
    col_name = 'Risk Factor' if 'Risk Factor' in results_df.columns else 'comparison'
    
    # We reconstruct the table row-by-row to guarantee the order
    # 1. Extract the BMI row
    row_bmi = results_df[results_df[col_name].str.contains("BMI", case=False, na=False)]
    
    # 2. Extract the Glucose row
    row_glucose = results_df[results_df[col_name].str.contains("Glucose", case=False, na=False)]
    
    # 3. Extract the 'Both' row
    row_both = results_df[results_df[col_name].str.contains("Both", case=False, na=False)]
    
    # Recombine them in this exact order: BMI first (index 0 -> Bottom), Both last (index 2 -> Top)
    plot_data = pd.concat([row_bmi, row_glucose, row_both], ignore_index=True)

    # --- Step 2: Plotting ---
    plt.figure(figsize=(10, 6))
    
    for i, row in plot_data.iterrows():
        # Significance Check
        is_significant = (row['CI_Lower'] > 1) or (row['CI_Upper'] < 1)
        color = '#d62828' if is_significant else 'black'
        
        # The Key Trick: Plot at position 'i' (0, 1, 2) based on our forced order
        plt.errorbar(x=row['RR'], y=i, 
                     xerr=[[row['RR']-row['CI_Lower']], [row['CI_Upper']-row['RR']]], 
                     fmt='o', color=color, ecolor=color, capsize=5, markersize=8)
        
        # Text Label above the point
        label_text = f"RR={row['RR']:.2f}\n({row['CI_Lower']:.2f}-{row['CI_Upper']:.2f})"
        plt.text(row['RR'], i + 0.15, label_text, va='bottom', ha='center', 
                 fontsize=9, fontweight='bold', color='darkblue')

    # Reference Line at 1
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='No Effect')
    
    # Set Y Ticks to the names in our manually sorted table
    plt.yticks(range(len(plot_data)), plot_data[col_name], fontsize=11, fontweight='bold')
    
    # Adjust limits for better spacing
    plt.ylim(-0.5, len(plot_data) - 0.5 + 0.5)

    # Custom Legend (Matching the Lollipop style)
    custom_lines = [
        Line2D([0], [0], color='#d62828', marker='o', linestyle='', markersize=8),
        Line2D([0], [0], color='black', marker='o', linestyle='', markersize=8)
    ]
    plt.legend(custom_lines, ['Significant', 'Not Significant'], loc='lower right', title="Significance")

    plt.title('Relative Risk Forest Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Risk (RR)', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("4_forest_plot.png")
    plt.close()
    logger.info("Saved: 4_forest_plot.png")

def plot_results_table(results_df, title):
    """
    Renders the results DataFrame as a static image table.
    This allows the table to appear in the 'Plots' pane of IDEs.
    """
    logger.info(f"Generating Results Table for: {title}")
    
    if results_df.empty:
        return

    # 1. Select and Rename Columns for cleaner display
    display_df = results_df[['comparison', 'RR', 'CI_Lower', 'CI_Upper', 'P_Value', 'Significant_0.05']].copy()
    display_df.columns = ['Group', 'RR', 'Lower CI', 'Upper CI', 'P-Value', 'Significant_0.05']
    
    # Round numbers for better visuals
    display_df['RR'] = display_df['RR'].round(2)
    display_df['Lower CI'] = display_df['Lower CI'].round(2)
    display_df['Upper CI'] = display_df['Upper CI'].round(2)
    display_df['P-Value'] = display_df['P-Value'].round(4)

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
        plot_rr_forest(results_df)
        plot_rr_lollipop(results_df)
    else:
        logger.warning("No results_df provided. Skipping RR plots.")
        
    logger.info("--- Visualization Pipeline Completed ---")
    