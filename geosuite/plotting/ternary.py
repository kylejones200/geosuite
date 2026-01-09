"""
Ternary plot creation for geoscience applications.

Ternary plots are used to visualize three-component data on a triangular diagram.
Common applications in geoscience include:
- Lithology classification (sand-silt-clay)
- Rock classification (Q-F-L, Q-F-R)
- Mineral composition
- Fluid composition

All plots use signalplot for consistent, minimalist styling.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import signalplot
import logging

logger = logging.getLogger(__name__)

# Apply signalplot style globally for this module
signalplot.apply()


def _ternary_to_cartesian(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ternary coordinates (a, b, c) to Cartesian (x, y).
    
    Ternary coordinates must sum to 1 (or 100 if using percentages).
    The triangle vertices are:
    - Top vertex: (0, 1) for component 'a'
    - Bottom-left: (0, 0) for component 'b'
    - Bottom-right: (1, 0) for component 'c'
    
    Args:
        a: First component (0-1 or 0-100)
        b: Second component (0-1 or 0-100)
        c: Third component (0-1 or 0-100)
        
    Returns:
        Tuple of (x, y) arrays in Cartesian coordinates
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    
    # Normalize if sum > 1.5 (assuming percentages)
    total = a + b + c
    if np.any(total > 1.5):
        a = a / total
        b = b / total
        c = c / total
    
    # Convert to Cartesian coordinates
    # x = (b + 2*c) / 2
    # y = (sqrt(3) * b) / 2
    x = (b + 2 * c) / 2.0
    y = (np.sqrt(3) * b) / 2.0
    
    return x, y


def _cartesian_to_ternary(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates (x, y) to ternary (a, b, c).
    
    Args:
        x: X coordinates (0-1)
        y: Y coordinates (0-1)
        
    Returns:
        Tuple of (a, b, c) arrays in ternary coordinates
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Convert from Cartesian to ternary
    b = (2 * y) / np.sqrt(3)
    c = x - (b / 2)
    a = 1 - b - c
    
    return a, b, c


def _setup_ternary_axes(
    ax: Axes,
    labels: Tuple[str, str, str] = ('A', 'B', 'C'),
    grid: bool = True,
    grid_style: str = 'dashed',
    grid_alpha: float = 0.3
) -> None:
    """
    Set up ternary plot axes with triangle frame and grid lines.
    
    Args:
        ax: Matplotlib axes
        labels: Tuple of (top, bottom-left, bottom-right) labels
        grid: If True, draw grid lines
        grid_style: Grid line style ('dashed', 'dotted', 'solid')
        grid_alpha: Grid line alpha (transparency)
    """
    # Clear axis
    ax.clear()
    
    # Draw triangle frame
    triangle_vertices = np.array([
        [0.5, np.sqrt(3) / 2],  # Top vertex
        [0, 0],                  # Bottom-left
        [1, 0]                   # Bottom-right
    ])
    
    triangle = mpatches.Polygon(triangle_vertices, closed=True, 
                                fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)
    
    # Add grid lines if requested
    if grid:
        # Grid lines parallel to each side (10 divisions)
        n_grid = 10
        
        # Lines parallel to bottom (horizontal)
        for i in range(1, n_grid):
            y_val = i * np.sqrt(3) / (2 * n_grid)
            x_start = (1 - (2 * i / n_grid)) / 2
            x_end = 1 - x_start
            ax.plot([x_start, x_end], [y_val, y_val], 
                   linestyle=grid_style, color='gray', alpha=grid_alpha, linewidth=0.5)
        
        # Lines parallel to right side (slanted left)
        for i in range(1, n_grid):
            # Bottom-left to right side
            y_start = 0
            x_start = i / n_grid
            y_end = np.sqrt(3) * (1 - i / n_grid) / 2
            x_end = 0.5 + i / (2 * n_grid)
            ax.plot([x_start, x_end], [y_start, y_end],
                   linestyle=grid_style, color='gray', alpha=grid_alpha, linewidth=0.5)
        
        # Lines parallel to left side (slanted right)
        for i in range(1, n_grid):
            # Bottom-right to left side
            y_start = 0
            x_start = 1 - i / n_grid
            y_end = np.sqrt(3) * (1 - i / n_grid) / 2
            x_end = 0.5 - i / (2 * n_grid)
            ax.plot([x_start, x_end], [y_start, y_end],
                   linestyle=grid_style, color='gray', alpha=grid_alpha, linewidth=0.5)
    
    # Add axis labels
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, labels[0], 
           ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(-0.05, -0.05, labels[1], 
           ha='right', va='top', fontsize=11, fontweight='bold')
    ax.text(1.05, -0.05, labels[2], 
           ha='left', va='top', fontsize=11, fontweight='bold')
    
    # Set axis limits with padding
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')


def ternary_plot(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    c_col: str,
    labels: Tuple[str, str, str] = None,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    title: str = 'Ternary Plot',
    figsize: Tuple[float, float] = (8, 8),
    grid: bool = True,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    s: float = 30
) -> Figure:
    """
    Create a ternary plot from three-component data.
    
    Args:
        df: DataFrame with three-component data
        a_col: Column name for first component (top vertex)
        b_col: Column name for second component (bottom-left)
        c_col: Column name for third component (bottom-right)
        labels: Tuple of (a_label, b_label, c_label). If None, uses column names.
        color_by: Optional column to color points by
        size_by: Optional column to size points by
        title: Plot title
        figsize: Figure size (width, height) in inches
        grid: If True, show grid lines
        cmap: Colormap for coloring
        alpha: Point transparency (0-1)
        s: Point size (if size_by not used)
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Sand': [0.5, 0.3, 0.7],
        ...     'Silt': [0.3, 0.5, 0.2],
        ...     'Clay': [0.2, 0.2, 0.1]
        ... })
        >>> fig = ternary_plot(df, 'Sand', 'Silt', 'Clay',
        ...                    labels=('Sand', 'Silt', 'Clay'),
        ...                    title='Texture Classification')
    """
    if labels is None:
        labels = (a_col, b_col, c_col)
    
    # Extract components
    a = df[a_col].values
    b = df[b_col].values
    c = df[c_col].values
    
    # Remove invalid data
    mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    a = a[mask]
    b = b[mask]
    c = c[mask]
    
    if len(a) == 0:
        raise ValueError("No valid data points after removing NaNs")
    
    # Convert to Cartesian coordinates
    x, y = _ternary_to_cartesian(a, b, c)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up ternary axes
    _setup_ternary_axes(ax, labels=labels, grid=grid)
    
    # Prepare point properties
    colors = None
    sizes = np.full(len(x), s)
    
    if color_by and color_by in df.columns:
        color_data = df[color_by].values[mask]
        colors = color_data
    
    if size_by and size_by in df.columns:
        size_data = df[size_by].values[mask]
        # Normalize sizes to reasonable range
        size_min, size_max = size_data.min(), size_data.max()
        if size_max > size_min:
            sizes = s * (0.5 + 1.5 * (size_data - size_min) / (size_max - size_min))
        else:
            sizes = np.full(len(x), s)
    
    # Plot points
    if colors is not None:
        scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=alpha, cmap=cmap, 
                           edgecolors='black', linewidths=0.5, zorder=3)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.05, shrink=0.8)
        cbar.set_label(color_by, fontsize=10)
    else:
        ax.scatter(x, y, s=sizes, alpha=alpha, color='black', 
                  edgecolors='white', linewidths=0.5, zorder=3)
    
    # Set title
    ax.set_title(title, fontsize=13, pad=20, fontweight='bold')
    
    plt.tight_layout()
    return fig


def sand_silt_clay_plot(
    df: pd.DataFrame,
    sand_col: str = 'Sand',
    silt_col: str = 'Silt',
    clay_col: str = 'Clay',
    color_by: Optional[str] = None,
    title: str = 'Sand-Silt-Clay Texture Classification',
    show_classification: bool = True,
    figsize: Tuple[float, float] = (10, 10),
    **kwargs
) -> Figure:
    """
    Create a sand-silt-clay ternary plot for soil/sediment texture classification.
    
    This is a specialized ternary plot with USDA texture classification zones.
    
    Args:
        df: DataFrame with sand, silt, clay percentages
        sand_col: Column name for sand percentage
        silt_col: Column name for silt percentage
        clay_col: Column name for clay percentage
        color_by: Optional column to color points by
        title: Plot title
        show_classification: If True, draw texture classification boundaries
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments passed to ternary_plot
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Sand': [50, 30, 70],
        ...     'Silt': [30, 50, 20],
        ...     'Clay': [20, 20, 10]
        ... })
        >>> fig = sand_silt_clay_plot(df, title='Sediment Texture')
    """
    fig = ternary_plot(
        df=df,
        a_col=sand_col,
        b_col=silt_col,
        c_col=clay_col,
        labels=('Sand', 'Silt', 'Clay'),
        title=title,
        figsize=figsize,
        color_by=color_by,
        **kwargs
    )
    
    # Add texture classification zones if requested
    if show_classification:
        # Note: Full USDA texture classification requires defining polygon vertices
        # for each texture class (e.g., Loam, Sandy Loam, Clay Loam, etc.)
        # This is a placeholder for future enhancement
        # Common texture classes boundaries can be added as polygon patches
        logger.debug("Texture classification zones can be enhanced by adding polygon patches")
    
    return fig


def qfl_plot(
    df: pd.DataFrame,
    q_col: str = 'Quartz',
    f_col: str = 'Feldspar',
    l_col: str = 'Lithics',
    color_by: Optional[str] = None,
    title: str = 'Q-F-L Rock Classification',
    figsize: Tuple[float, float] = (10, 10),
    **kwargs
) -> Figure:
    """
    Create a Q-F-L (Quartz-Feldspar-Lithics) ternary plot for rock classification.
    
    This ternary diagram is commonly used in sedimentary petrology to classify
    sandstones and other clastic rocks.
    
    Args:
        df: DataFrame with Q, F, L percentages
        q_col: Column name for Quartz percentage
        f_col: Column name for Feldspar percentage
        l_col: Column name for Lithics percentage
        color_by: Optional column to color points by
        title: Plot title
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments passed to ternary_plot
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Quartz': [60, 40, 80],
        ...     'Feldspar': [25, 40, 10],
        ...     'Lithics': [15, 20, 10]
        ... })
        >>> fig = qfl_plot(df, title='Sandstone Classification')
    """
    return ternary_plot(
        df=df,
        a_col=q_col,
        b_col=f_col,
        c_col=l_col,
        labels=('Quartz', 'Feldspar', 'Lithics'),
        title=title,
        figsize=figsize,
        color_by=color_by,
        **kwargs
    )


def mineral_composition_plot(
    df: pd.DataFrame,
    mineral1_col: str,
    mineral2_col: str,
    mineral3_col: str,
    labels: Tuple[str, str, str] = None,
    color_by: Optional[str] = None,
    title: str = 'Mineral Composition',
    figsize: Tuple[float, float] = (10, 10),
    **kwargs
) -> Figure:
    """
    Create a ternary plot for mineral composition analysis.
    
    General-purpose ternary plot for any three-component mineral composition.
    
    Args:
        df: DataFrame with mineral percentages
        mineral1_col: Column name for first mineral
        mineral2_col: Column name for second mineral
        mineral3_col: Column name for third mineral
        labels: Tuple of mineral labels. If None, uses column names.
        color_by: Optional column to color points by
        title: Plot title
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments passed to ternary_plot
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Quartz': [40, 50, 60],
        ...     'Feldspar': [30, 25, 20],
        ...     'Mica': [30, 25, 20]
        ... })
        >>> fig = mineral_composition_plot(df, 'Quartz', 'Feldspar', 'Mica',
        ...                                title='Rock Mineralogy')
    """
    if labels is None:
        labels = (mineral1_col, mineral2_col, mineral3_col)
    
    return ternary_plot(
        df=df,
        a_col=mineral1_col,
        b_col=mineral2_col,
        c_col=mineral3_col,
        labels=labels,
        title=title,
        figsize=figsize,
        color_by=color_by,
        **kwargs
    )

