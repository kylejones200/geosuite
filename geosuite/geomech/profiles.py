"""
Generate depth profiles for pressure and stress visualization.

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


def plot_pressure_profile(
    df: pd.DataFrame,
    depth_col: str = 'Depth',
    pressure_cols: Optional[list] = None,
    title: str = 'Pressure Profile',
    depth_units: str = 'm',
    pressure_units: str = 'MPa',
    figsize: Tuple[float, float] = (6, 8)
) -> Figure:
    """
    Create pressure profile plot.
    
    Args:
        df: DataFrame with depth and pressure data
        depth_col: Name of depth column
        pressure_cols: List of pressure column names (e.g., ['Sv', 'Ph', 'Pp'])
        title: Plot title
        depth_units: Units for depth axis
        pressure_units: Units for pressure axis
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    if pressure_cols is None:
        pressure_cols = ['Sv', 'Ph', 'Pp']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {
        'Sv': 'black',
        'Ph': 'blue',
        'Pp': 'red',
        'Shmin': 'green',
        'SHmax': 'orange',
        'Sigma_eff': 'purple'
    }
    
    linestyles = {
        'Sv': '-',
        'Ph': '--',
        'Pp': '-',
        'Shmin': '-.',
        'SHmax': ':',
        'Sigma_eff': '--'
    }
    
    for col in pressure_cols:
        if col in df.columns:
            ax.plot(df[col], df[depth_col],
                   color=colors.get(col, 'gray'),
                   linestyle=linestyles.get(col, '-'),
                   linewidth=1.5,
                   label=col)
    
    # Labels and title
    ax.set_xlabel(f'Pressure ({pressure_units})')
    ax.set_ylabel(f'Depth ({depth_units})')
    ax.set_title(title)
    
    # Invert y-axis for depth (geological convention)
    ax.invert_yaxis()
    
    # Legend
    ax.legend(loc='best')
    
    # signalplot handles spines automatically
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    return fig


def plot_mud_weight_profile(
    df: pd.DataFrame,
    depth_col: str = 'Depth',
    mw_cols: Optional[list] = None,
    title: str = 'Equivalent Mud Weight',
    depth_units: str = 'm',
    figsize: Tuple[float, float] = (6, 8)
) -> Figure:
    """
    Create equivalent mud weight profile plot.
    
    Args:
        df: DataFrame with depth and mud weight data
        depth_col: Name of depth column
        mw_cols: List of mud weight column names
        title: Plot title
        depth_units: Units for depth axis
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    if mw_cols is None:
        mw_cols = [col for col in df.columns if 'MW' in col or 'mud_weight' in col.lower()]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in mw_cols:
        if col in df.columns:
            ax.plot(df[col], df[depth_col], linewidth=1.5, label=col)
    
    # Labels and title
    ax.set_xlabel('Mud Weight (g/cc)', fontsize=11)
    ax.set_ylabel(f'Depth ({depth_units})', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    
    # Invert y-axis for depth
    ax.invert_yaxis()
    
    # Legend
    ax.legend(frameon=False, fontsize=9, loc='best')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    return fig
