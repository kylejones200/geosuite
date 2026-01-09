"""
Block model generation for mining applications.

This module provides functions for creating 3D block models from drillhole data,
including grid generation, grade estimation, and export to industry-standard formats.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_block_model_grid(
    coords: np.ndarray,
    block_size_xy: float = 25.0,
    block_size_z: float = 10.0,
    bounds: Optional[Dict[str, float]] = None,
    quantile_padding: float = 0.05
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a 3D block model grid from sample coordinates.
    
    Generates a regular 3D grid with specified block sizes, suitable for
    mine planning applications. Grid bounds can be specified or auto-computed
    from data with optional padding.
    
    Args:
        coords: Sample coordinates (n_samples, 3) - [X, Y, Z]
        block_size_xy: Block size in X and Y directions (meters)
        block_size_z: Block size in Z direction (meters)
        bounds: Optional dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max',
                'z_min', 'z_max'. If None, computed from data.
        quantile_padding: Padding as quantile (0-1) if bounds not specified.
                         Uses 5th and 95th percentiles by default.
        
    Returns:
        Tuple of (grid_coords, grid_info):
            - grid_coords: Grid coordinates (n_blocks, 3) - block centroids
            - grid_info: Dictionary with grid metadata:
                - 'nx', 'ny', 'nz': Grid dimensions
                - 'n_blocks': Total number of blocks
                - 'x_range', 'y_range', 'z_range': Coordinate ranges
                - 'block_size_xy', 'block_size_z': Block sizes
                
    Example:
        >>> coords = samples[['x', 'y', 'z']].values
        >>> grid, info = create_block_model_grid(coords, block_size_xy=25, block_size_z=10)
        >>> print(f"Grid: {info['nx']} × {info['ny']} × {info['nz']} = {info['n_blocks']:,} blocks")
    """
    coords = np.asarray(coords, dtype=np.float64)
    
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be 2D array with 3 columns (X, Y, Z)")
    
    if len(coords) == 0:
        raise ValueError("Coordinates cannot be empty")
    
    # Determine bounds
    if bounds is None:
        # Use quantiles to exclude outliers
        x_min, x_max = np.quantile(coords[:, 0], [quantile_padding, 1 - quantile_padding])
        y_min, y_max = np.quantile(coords[:, 1], [quantile_padding, 1 - quantile_padding])
        z_min, z_max = np.quantile(coords[:, 2], [quantile_padding, 1 - quantile_padding])
    else:
        x_min = bounds['x_min']
        x_max = bounds['x_max']
        y_min = bounds['y_min']
        y_max = bounds['y_max']
        z_min = bounds['z_min']
        z_max = bounds['z_max']
    
    # Calculate grid dimensions
    nx = int(np.ceil((x_max - x_min) / block_size_xy))
    ny = int(np.ceil((y_max - y_min) / block_size_xy))
    nz = int(np.ceil((z_max - z_min) / block_size_z))
    
    # Adjust bounds to be evenly divisible by block size
    x_max = x_min + nx * block_size_xy
    y_max = y_min + ny * block_size_xy
    z_max = z_min + nz * block_size_z
    
    # Create grid coordinates (block centroids)
    x_coords = np.linspace(x_min, x_max, nx) + block_size_xy / 2
    y_coords = np.linspace(y_min, y_max, ny) + block_size_xy / 2
    z_coords = np.linspace(z_min, z_max, nz) + block_size_z / 2
    
    # Create 3D meshgrid
    G_x, G_y, G_z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Flatten to coordinate matrix
    grid_coords = np.column_stack([
        G_x.ravel(),
        G_y.ravel(),
        G_z.ravel()
    ])
    
    n_blocks = len(grid_coords)
    
    grid_info = {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'n_blocks': n_blocks,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'z_range': (z_min, z_max),
        'block_size_xy': block_size_xy,
        'block_size_z': block_size_z
    }
    
    logger.info(
        f"Created block model grid: {nx} × {ny} × {nz} = {n_blocks:,} blocks"
    )
    
    return grid_coords, grid_info


def export_block_model(
    block_model: pd.DataFrame,
    filename: Union[str, Path],
    format: str = 'csv',
    include_metadata: bool = True
) -> None:
    """
    Export block model to file.
    
    Supports CSV format (compatible with Vulcan, Datamine, Leapfrog, Surpac).
    Standard columns include X, Y, Z coordinates and grade estimates.
    
    Args:
        block_model: DataFrame with block model data
        filename: Output file path
        format: Export format ('csv', 'parquet'). Default 'csv' for compatibility.
        include_metadata: If True, add comment lines with metadata
        
    Example:
        >>> block_model = pd.DataFrame({
        ...     'x': [100, 125, 150],
        ...     'y': [200, 200, 200],
        ...     'z': [50, 50, 50],
        ...     'grade': [2.5, 1.8, 2.1]
        ... })
        >>> export_block_model(block_model, 'block_model.csv')
    """
    filename = Path(filename)
    
    if format == 'csv':
        # CSV export with optional metadata
        if include_metadata:
            with open(filename, 'w') as f:
                f.write(f"# Block Model Export\n")
                f.write(f"# Total blocks: {len(block_model)}\n")
                f.write(f"# Columns: {', '.join(block_model.columns)}\n")
                f.write(f"#\n")
            # Append CSV data
            block_model.to_csv(filename, mode='a', index=False)
        else:
            block_model.to_csv(filename, index=False)
        
        logger.info(f"Exported block model to {filename} ({len(block_model):,} blocks)")
    
    elif format == 'parquet':
        block_model.to_parquet(filename, index=False)
        logger.info(f"Exported block model to {filename} ({len(block_model):,} blocks)")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'")

