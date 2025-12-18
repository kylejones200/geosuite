"""
Buckles plot for bulk volume water analysis.

All plots use signalplot for consistent, minimalist styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import signalplot
from typing import Optional, Tuple
from matplotlib.figure import Figure

# Apply signalplot style globally for this module
signalplot.apply()


def buckles_plot(
    df: pd.DataFrame,
    porosity_col: str = 'PHIND',
    sw_col: str = 'SW',
    cutoff: float = 0.04,
    title: str = 'Buckles Plot',
    color_by: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> Figure:
    """
    Create a Buckles plot for reservoir quality analysis.
    
    The Buckles plot (Bulk Volume Water vs Porosity) is used to identify
    productive reservoir zones. The BVW cutoff separates water-bearing
    from hydrocarbon-bearing zones.
    
    Args:
        df: DataFrame with petrophysical data
        porosity_col: Column name for porosity
        sw_col: Column name for water saturation
        cutoff: BVW cutoff value (typically 0.03-0.05)
        title: Plot title
        color_by: Optional column to color points by
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Calculate Bulk Volume Water
    df_plot = df[[porosity_col, sw_col]].dropna()
    df_plot = df_plot[(df_plot[porosity_col] > 0) & (df_plot[sw_col] > 0)]
    
    phi = df_plot[porosity_col].values
    sw = df_plot[sw_col].values
    bvw = phi * sw
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points
    if color_by and color_by in df.columns:
        color_data = df.loc[df_plot.index, color_by]
        scatter = ax.scatter(phi, bvw, c=color_data, s=30, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        scatter = ax.scatter(phi, bvw, c=bvw, s=30, alpha=0.6, cmap='RdYlGn_r')
        plt.colorbar(scatter, ax=ax, label='BVW')
    
    # Add cutoff line
    phi_range = np.linspace(0, phi.max() * 1.1, 100)
    ax.axhline(y=cutoff, color='black', linestyle='--', linewidth=1.5, 
               label=f'BVW Cutoff = {cutoff}')
    
    # Add Sw contours
    for sw_line in [0.2, 0.4, 0.6, 0.8, 1.0]:
        bvw_line = phi_range * sw_line
        ax.plot(phi_range, bvw_line, 'k:', linewidth=0.8, alpha=0.5)
        # Add label at the end of the line
        if sw_line in [0.2, 0.6, 1.0]:
            ax.text(phi.max() * 1.05, phi.max() * sw_line * 1.05, 
                   f'Sw={sw_line}', fontsize=8, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Porosity (v/v)', fontsize=11)
    ax.set_ylabel('Bulk Volume Water (BVW)', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    
    # Set axis limits
    ax.set_xlim(0, phi.max() * 1.1)
    ax.set_ylim(0, max(bvw.max() * 1.1, cutoff * 2))
    
    # Add zone labels
    y_mid = ax.get_ylim()[1]
    ax.text(phi.max() * 0.5, cutoff * 0.5, 'Hydrocarbon Zone',
            ha='center', va='center', fontsize=10, color='green', alpha=0.7)
    ax.text(phi.max() * 0.5, cutoff * 1.5, 'Water Zone',
            ha='center', va='center', fontsize=10, color='red', alpha=0.7)
    
    # Legend
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig
