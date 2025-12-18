"""
Lithology identification crossplots (neutron-density, etc.).

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


def neutron_density_crossplot(
    df: pd.DataFrame,
    nphi_col: str = 'NPHI',
    rhob_col: str = 'RHOB',
    color_by: Optional[str] = None,
    title: str = 'Neutron-Density Crossplot',
    show_lithology_lines: bool = True,
    figsize: Tuple[float, float] = (8, 6)
) -> Figure:
    """
    Create neutron-density crossplot for lithology identification.
    
    This plot helps identify different rock types and fluid content
    based on their neutron and density log responses.
    
    Args:
        df: DataFrame with log data
        nphi_col: Neutron porosity column
        rhob_col: Bulk density column
        color_by: Optional column to color points by
        title: Plot title
        show_lithology_lines: If True, show lithology reference lines
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Remove invalid data
    df_plot = df[[nphi_col, rhob_col]].dropna()
    
    nphi = df_plot[nphi_col].values
    rhob = df_plot[rhob_col].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add lithology reference lines first (behind data)
    if show_lithology_lines:
        # Sandstone line (typical values)
        ss_phi = np.linspace(0, 0.4, 50)
        ss_rhob = 2.65 - ss_phi * (2.65 - 1.0)  # sandstone matrix = 2.65
        ax.plot(ss_phi, ss_rhob, '-', color='gold', linewidth=2, label='Sandstone', alpha=0.7)
        
        # Limestone line
        ls_phi = np.linspace(0, 0.4, 50)
        ls_rhob = 2.71 - ls_phi * (2.71 - 1.0)  # limestone matrix = 2.71
        ax.plot(ls_phi, ls_rhob, '-', color='gray', linewidth=2, label='Limestone', alpha=0.7)
        
        # Dolomite line
        dol_phi = np.linspace(0, 0.4, 50)
        dol_rhob = 2.87 - dol_phi * (2.87 - 1.0)  # dolomite matrix = 2.87
        ax.plot(dol_phi, dol_rhob, '-', color='saddlebrown', linewidth=2, label='Dolomite', alpha=0.7)
    
    # Plot data points
    if color_by and color_by in df.columns:
        color_data = df.loc[df_plot.index, color_by]
        scatter = ax.scatter(nphi, rhob, c=color_data, s=20, alpha=0.6, cmap='viridis', zorder=3)
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(nphi, rhob, s=20, alpha=0.6, color='black', zorder=3)
    
    # Labels and title
    ax.set_xlabel('Neutron Porosity (v/v)', fontsize=11)
    ax.set_ylabel('Bulk Density (g/cc)', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    
    # Set axis limits and invert y-axis
    ax.set_xlim(-0.05, 0.45)
    ax.set_ylim(3.0, 1.8)  # Inverted
    
    # Legend
    if show_lithology_lines:
        ax.legend(frameon=False, fontsize=9, loc='upper right')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig
