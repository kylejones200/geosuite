"""
Strip chart (well log plot) creation functions.

This module provides functions to create strip charts for
visualizing well log data in the traditional multi-track format.

All plots use signalplot for consistent, minimalist styling.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Union, Tuple

# Apply signalplot style globally for this module
signalplot.apply()

logger = logging.getLogger(__name__)


# Default color schemes for facies
FACIES_COLORS = {
    'Sand': '#FFFF00',
    'Shale': '#808080',
    'Siltstone': '#C8C896',
    'Carbonate': '#0000FF',
    'Limestone': '#87CEEB',
    'Dolomite': '#FFB6C1',
    'Coal': '#000000',
    'Clean_Sand': '#FFFF00',
    'Shaly_Sand': '#F0E68C',
    'Mudstone': '#8B4513'
}

# Common log display settings
LOG_SETTINGS = {
    'GR': {
        'name': 'Gamma Ray',
        'unit': 'API',
        'color': 'green',
        'range': [0, 150]
    },
    'RHOB': {
        'name': 'Bulk Density',
        'unit': 'g/cc',
        'color': 'red',
        'range': [1.8, 2.8]
    },
    'NPHI': {
        'name': 'Neutron Porosity',
        'unit': 'v/v',
        'color': 'blue',
        'range': [0.45, -0.15]  # Reversed scale
    },
    'RT': {
        'name': 'Resistivity',
        'unit': 'ohm.m',
        'color': 'black',
        'range': [0.2, 2000],
        'log_scale': True
    },
    'ILD': {
        'name': 'Deep Resistivity',
        'unit': 'ohm.m',
        'color': 'black',
        'range': [0.2, 2000],
        'log_scale': True
    },
    'PE': {
        'name': 'Photo Electric',
        'unit': 'b/e',
        'color': 'purple',
        'range': [0, 10]
    },
    'DT': {
        'name': 'Sonic',
        'unit': 'us/ft',
        'color': 'blue',
        'range': [140, 40]  # Reversed scale
    },
    'CALI': {
        'name': 'Caliper',
        'unit': 'in',
        'color': 'orange',
        'range': [6, 16]
    }
}


def create_strip_chart(
    df: pd.DataFrame,
    depth_col: str = 'DEPTH',
    log_cols: Optional[List[str]] = None,
    facies_col: Optional[str] = None,
    title: str = 'Well Log Strip Chart',
    figsize: Optional[Tuple[float, float]] = None,
    depth_range: Optional[tuple] = None,
    colors: Optional[Dict[str, str]] = None
) -> Figure:
    """
    Create a strip chart (well log plot).
    
    Args:
        df: DataFrame with well log data
        depth_col: Name of depth column
        log_cols: List of log column names to plot. If None, uses common logs.
        facies_col: Optional facies column name for facies track
        title: Plot title
        figsize: Figure size (width, height) in inches. If None, auto-sized.
        depth_range: Optional (min, max) depth range to display
        colors: Optional dictionary mapping log names to colors
        
    Returns:
        Matplotlib Figure object
    """
    # Determine which logs to plot
    if log_cols is None:
        # Auto-detect available common logs
        common_logs = ['GR', 'RHOB', 'NPHI', 'RT', 'ILD', 'PE', 'DT', 'CALI']
        log_cols = [col for col in common_logs if col in df.columns]
        
        if not log_cols:
            # Fallback to any numeric columns except depth
            log_cols = [col for col in df.columns 
                       if col != depth_col and pd.api.types.is_numeric_dtype(df[col])][:4]
    
    # Apply depth range filter if specified
    if depth_range:
        df_plot = df[(df[depth_col] >= depth_range[0]) & 
                     (df[depth_col] <= depth_range[1])].copy()
    else:
        df_plot = df.copy()
    
    # Add facies track if specified
    n_tracks = len(log_cols) + (1 if facies_col else 0)
    
    # Auto-size if not specified
    if figsize is None:
        width = min(2.5 * n_tracks, 15)
        height = 10
        figsize = (width, height)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_tracks, figsize=figsize, sharey=True)
    
    # Handle single track case
    if n_tracks == 1:
        axes = [axes]
    
    # Plot each log track
    for idx, col in enumerate(log_cols):
        ax = axes[idx]
        add_log_track(ax, df_plot, depth_col, col, colors)
    
    # Add facies track if specified
    if facies_col:
        ax = axes[-1]
        add_facies_track(ax, df_plot, depth_col, facies_col)
    
    # Set overall title
    fig.suptitle(title, fontsize=13, y=0.995)
    
    plt.tight_layout()
    return fig


def add_log_track(
    ax: plt.Axes,
    df: pd.DataFrame,
    depth_col: str,
    log_col: str,
    colors: Optional[Dict[str, str]] = None
) -> None:
    """
    Add a single log track to an axis.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with log data
        depth_col: Name of depth column
        log_col: Name of log column to plot
        colors: Optional dictionary mapping log names to colors
    """
    # Get settings for this log
    settings = LOG_SETTINGS.get(log_col, {})
    log_name = settings.get('name', log_col)
    unit = settings.get('unit', '')
    color = colors.get(log_col) if colors else settings.get('color', 'black')
    log_range = settings.get('range')
    log_scale = settings.get('log_scale', False)
    
    # Plot the log
    depth = df[depth_col].values
    log_data = df[log_col].values
    
    ax.plot(log_data, depth, color=color, linewidth=1)
    
    # Set axis properties
    if log_scale:
        ax.set_xscale('log')
    
    if log_range:
        ax.set_xlim(log_range)
    
    ax.set_xlabel(f'{log_name}\n({unit})' if unit else log_name)
    
    # Invert y-axis for depth (geological convention)
    ax.invert_yaxis()
    
    # signalplot handles spines and grid automatically


def add_facies_track(
    ax: plt.Axes,
    df: pd.DataFrame,
    depth_col: str,
    facies_col: str,
    facies_colors: Optional[Dict] = None
) -> None:
    """
    Add a facies track to an axis.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with facies data
        depth_col: Name of depth column
        facies_col: Name of facies column
        facies_colors: Optional dictionary mapping facies to colors
    """
    if facies_colors is None:
        facies_colors = FACIES_COLORS
    
    depth = df[depth_col].values
    facies = df[facies_col].values
    
    # Get unique facies
    unique_facies = df[facies_col].unique()
    
    # Plot facies as colored bands
    for i in range(len(depth) - 1):
        facies_name = facies[i]
        color = facies_colors.get(facies_name, '#CCCCCC')
        
        ax.fill_betweenx([depth[i], depth[i+1]], 0, 1, 
                         color=color, alpha=0.8)
    
    ax.set_xlabel('Facies')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    
    # Add legend
    handles = []
    labels = []
    for facies_name in sorted(unique_facies):
        if pd.notna(facies_name):
            color = facies_colors.get(facies_name, '#CCCCCC')
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8))
            labels.append(str(facies_name))
    
    ax.legend(handles, labels, loc='upper right', frameon=False, bbox_to_anchor=(1.0, 1.0))
    
    # signalplot handles spines automatically


def create_facies_log_plot(
    df: pd.DataFrame,
    depth_col: str = 'DEPTH',
    facies_col: str = 'Facies',
    log_cols: Optional[List[str]] = None,
    title: str = 'Well Logs with Facies',
    figsize: Optional[Tuple[float, float]] = None,
    depth_range: Optional[tuple] = None
) -> Figure:
    """
    Create a strip chart with facies track.
    
    This is a convenience function that ensures facies are displayed.
    
    Args:
        df: DataFrame with well log and facies data
        depth_col: Name of depth column
        facies_col: Name of facies column
        log_cols: List of log column names to plot
        title: Plot title
        figsize: Figure size (width, height) in inches
        depth_range: Optional (min, max) depth range to display
        
    Returns:
        Matplotlib Figure object
    """
    return create_strip_chart(
        df=df,
        depth_col=depth_col,
        log_cols=log_cols,
        facies_col=facies_col,
        title=title,
        figsize=figsize,
        depth_range=depth_range
    )


def create_multi_well_strip_chart(
    dfs: List[pd.DataFrame],
    well_names: List[str],
    depth_col: str = 'DEPTH',
    log_cols: Optional[List[str]] = None,
    facies_col: Optional[str] = None,
    title: str = 'Multi-Well Strip Chart',
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Create strip charts for multiple wells side-by-side.
    
    Args:
        dfs: List of DataFrames, one per well
        well_names: List of well names
        depth_col: Name of depth column
        log_cols: List of log column names to plot
        facies_col: Optional facies column name
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    n_wells = len(dfs)
    
    if log_cols is None:
        # Auto-detect from first well
        common_logs = ['GR', 'RHOB', 'NPHI', 'RT']
        log_cols = [col for col in common_logs if col in dfs[0].columns]
    
    n_tracks_per_well = len(log_cols) + (1 if facies_col else 0)
    
    # Auto-size if not specified
    if figsize is None:
        width = min(2.5 * n_tracks_per_well * n_wells, 20)
        height = 10
        figsize = (width, height)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    for well_idx, (df, well_name) in enumerate(zip(dfs, well_names)):
        # Create subplots for this well
        for track_idx in range(n_tracks_per_well):
            ax_idx = well_idx * n_tracks_per_well + track_idx + 1
            ax = fig.add_subplot(1, n_wells * n_tracks_per_well, ax_idx, sharey=(ax_idx > 1))
            
            if track_idx < len(log_cols):
                # Log track
                add_log_track(ax, df, depth_col, log_cols[track_idx], None)
                
                # Add well name to first track
                if track_idx == 0:
                    ax.text(0.5, 1.02, well_name, transform=ax.transAxes,
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            elif facies_col:
                # Facies track
                add_facies_track(ax, df, depth_col, facies_col)
            
            # Only show y-label on leftmost plot
            if well_idx > 0:
                ax.set_ylabel('')
    
    fig.suptitle(title, fontsize=13, y=0.995)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Strip chart module loaded")
    logger.info("Example: create_strip_chart(df, depth_col='DEPTH', log_cols=['GR', 'RHOB'])")
