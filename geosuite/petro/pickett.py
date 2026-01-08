"""
Pickett plot creation for porosity-resistivity analysis.

All plots use signalplot for consistent, minimalist styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import signalplot
from typing import Optional, Tuple, TYPE_CHECKING
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..config import ConfigManager
else:
    ConfigManager = None

# Apply signalplot style globally for this module
signalplot.apply()


def pickett_plot(
    df: pd.DataFrame,
    porosity_col: str = 'NPHI',
    resistivity_col: str = 'RT',
    m: Optional[float] = None,
    n: Optional[float] = None,
    a: Optional[float] = None,
    rw: Optional[float] = None,
    color_by: Optional[str] = None,
    title: str = 'Pickett Plot',
    show_water_line: bool = True,
    show_sw_lines: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    config: Optional['ConfigManager'] = None
) -> Figure:
    """
    Create a Pickett plot for porosity and resistivity analysis.
    
    The Pickett plot is a log-log crossplot of porosity vs resistivity
    used to determine water saturation and identify reservoir quality.
    
    Args:
        df: DataFrame with well log data
        porosity_col: Column name for porosity (fraction, not %)
        resistivity_col: Column name for resistivity
        m: Cementation exponent (typically 1.8-2.5)
        n: Saturation exponent (typically ~2)
        a: Tortuosity factor (typically 0.6-1.4)
        rw: Formation water resistivity (ohm-m)
        color_by: Optional column to color points by
        title: Plot title
        show_water_line: If True, show 100% water saturation line
        show_sw_lines: If True, show Sw = 50%, 25% lines
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Load from config if not provided
    if m is None:
        from ..config import get_config
        m = config.get("petro.archie.m", 2.0) if config else get_config("petro.archie.m", 2.0)
    if n is None:
        from ..config import get_config
        n = config.get("petro.archie.n", 2.0) if config else get_config("petro.archie.n", 2.0)
    if a is None:
        from ..config import get_config
        a = config.get("petro.archie.a", 1.0) if config else get_config("petro.archie.a", 1.0)
    if rw is None:
        from ..config import get_config
        rw = config.get("petro.archie.rw", 0.05) if config else get_config("petro.archie.rw", 0.05)
    
    # Remove invalid data
    df_plot = df[[porosity_col, resistivity_col]].dropna()
    df_plot = df_plot[(df_plot[porosity_col] > 0) & (df_plot[resistivity_col] > 0)]
    
    phi = df_plot[porosity_col].values
    rt = df_plot[resistivity_col].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points
    if color_by and color_by in df.columns:
        color_data = df.loc[df_plot.index, color_by]
        scatter = ax.scatter(phi, rt, c=color_data, s=20, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(phi, rt, s=20, alpha=0.6, color='black')
    
    # Add water saturation lines
    if show_water_line or show_sw_lines:
        phi_range = np.logspace(np.log10(phi.min()), np.log10(phi.max()), 100)
        
        if show_water_line:
            rt_100 = (a * rw) / (phi_range ** m)
            ax.plot(phi_range, rt_100, 'k--', linewidth=1.5, label='Sw = 100%')
        
        if show_sw_lines:
            rt_50 = (a * rw) / ((0.5 ** n) * (phi_range ** m))
            ax.plot(phi_range, rt_50, 'k:', linewidth=1, label='Sw = 50%')
            
            rt_25 = (a * rw) / ((0.25 ** n) * (phi_range ** m))
            ax.plot(phi_range, rt_25, 'k:', linewidth=1, label='Sw = 25%', alpha=0.7)
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Porosity (v/v)')
    ax.set_ylabel('Resistivity (ohm-m)')
    ax.set_title(title)
    
    # Legend
    if show_water_line or show_sw_lines:
        ax.legend(loc='best')
    
    # Add parameter text
    param_text = f'm={m}, n={n}, a={a}, Rw={rw}'
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # signalplot handles spines automatically
    
    plt.tight_layout()
    return fig
