"""
Feature engineering for ore geomodeling.

This module provides functions for creating spatial and geological features
from drillhole data for machine learning models.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.neighbors import KDTree
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available. Feature engineering requires scikit-learn. "
        "Install with: pip install scikit-learn"
    )


def build_spatial_features(
    coords: np.ndarray,
    return_scalers: bool = False,
    include_depth: bool = True,
    include_polynomial: bool = True,
    poly_degree: int = 2,
    include_density: bool = True,
    k_neighbors: int = 8
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Build spatial features from 3D coordinates for machine learning models.
    
    Creates a feature matrix with:
    - Normalized spatial coordinates (X, Y, Z)
    - Local point density (distance to k-th nearest neighbor)
    - Depth proxy (negative Z, useful for depth trends)
    - Polynomial features (X^2, Y^2, X*Y, etc.)
    
    Args:
        coords: Coordinates array (n_samples, 3) - [X, Y, Z]
        return_scalers: If True, return fitted scalers for reuse on grid
        include_depth: If True, include depth proxy feature
        include_polynomial: If True, include polynomial features (X^2, Y^2, X*Y)
        poly_degree: Degree of polynomial features (2 for quadratic)
        include_density: If True, include local point density feature
        k_neighbors: Number of neighbors for density calculation
        
    Returns:
        Feature matrix (n_samples, n_features), or tuple of
        (features, scalers_dict) if return_scalers=True
        
    Example:
        >>> coords = samples[['x', 'y', 'z']].values
        >>> features = build_spatial_features(coords)
        >>> print(f"Feature matrix shape: {features.shape}")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature engineering")
    
    coords = np.asarray(coords, dtype=np.float64)
    
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be 2D array with 3 columns (X, Y, Z)")
    
    if len(coords) == 0:
        raise ValueError("Coordinates cannot be empty")
    
    features_list = []
    scalers_dict = {}
    
    # 1. Normalized spatial coordinates
    scaler_coords = StandardScaler()
    coords_norm = scaler_coords.fit_transform(coords)
    features_list.append(coords_norm)
    
    if return_scalers:
        scalers_dict['coords_scaler'] = scaler_coords
    
    # 2. Local point density (distance to k-th nearest neighbor)
    if include_density:
        tree = KDTree(coords)
        k_actual = min(k_neighbors + 1, len(coords))  # +1 because first neighbor is self
        distances_nn, _ = tree.query(coords, k=k_actual)
        
        if distances_nn.ndim == 1:
            # Single neighbor case
            nn_dist = distances_nn
        else:
            # Get k-th neighbor distance (last column)
            nn_dist = distances_nn[:, -1]
        
        # Normalize density
        if nn_dist.std() > 1e-9:
            nn_dist = (nn_dist - nn_dist.mean()) / nn_dist.std()
        
        features_list.append(nn_dist.reshape(-1, 1))
    
    # 3. Depth proxy (negative Z, useful if mineralization varies with depth)
    if include_depth:
        depth_proxy = -coords[:, 2]  # Negative Z = deeper
        # Normalize
        if depth_proxy.std() > 1e-9:
            depth_proxy = (depth_proxy - depth_proxy.mean()) / depth_proxy.std()
        features_list.append(depth_proxy.reshape(-1, 1))
    
    # 4. Polynomial features (for capturing regional trends)
    if include_polynomial:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        coords_poly = poly.fit_transform(coords[:, :2])  # Only X, Y (not Z)
        
        # Exclude linear terms (already in coords_norm)
        # Keep only quadratic and higher terms
        if poly_degree >= 2:
            # Find indices of polynomial terms (X^2, Y^2, X*Y, etc.)
            # Linear terms are first 2, so skip them
            poly_features = coords_poly[:, 2:]  # Skip X, Y (already normalized)
            features_list.append(poly_features)
        
        if return_scalers:
            scalers_dict['poly_features'] = poly
    
    # Combine all features
    features = np.column_stack(features_list)
    
    logger.info(f"Built {features.shape[1]} spatial features from {len(coords)} samples")
    
    if return_scalers:
        return features, scalers_dict
    else:
        return features


def build_block_model_features(
    grid_coords: np.ndarray,
    sample_coords: np.ndarray,
    scalers_dict: Dict[str, Any],
    include_depth: bool = True,
    include_polynomial: bool = True,
    include_density: bool = True,
    k_neighbors: int = 8
) -> np.ndarray:
    """
    Build features for block model grid using fitted scalers.
    
    Uses the same feature engineering as build_spatial_features but applies
    pre-fitted scalers from sample data to grid coordinates.
    
    Args:
        grid_coords: Grid coordinates (n_blocks, 3) - [X, Y, Z]
        sample_coords: Sample coordinates used for fitting scalers (n_samples, 3)
        scalers_dict: Dictionary of fitted scalers from build_spatial_features
                     with return_scalers=True
        include_depth: If True, include depth proxy feature
        include_polynomial: If True, include polynomial features
        include_density: If True, include local point density
        k_neighbors: Number of neighbors for density calculation
        
    Returns:
        Feature matrix (n_blocks, n_features)
        
    Example:
        >>> # Build features on samples
        >>> features, scalers = build_spatial_features(sample_coords, return_scalers=True)
        >>> # Build features on grid using same scalers
        >>> grid_features = build_block_model_features(grid_coords, sample_coords, scalers)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature engineering")
    
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    sample_coords = np.asarray(sample_coords, dtype=np.float64)
    
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError("Grid coordinates must be 2D array with 3 columns")
    
    if sample_coords.ndim != 2 or sample_coords.shape[1] != 3:
        raise ValueError("Sample coordinates must be 2D array with 3 columns")
    
    if 'coords_scaler' not in scalers_dict:
        raise ValueError("scalers_dict must contain 'coords_scaler' from build_spatial_features")
    
    features_list = []
    
    # 1. Normalized spatial coordinates (using fitted scaler)
    scaler_coords = scalers_dict['coords_scaler']
    grid_norm = scaler_coords.transform(grid_coords)
    features_list.append(grid_norm)
    
    # 2. Local density at grid points (distance to nearest sample)
    if include_density:
        tree = KDTree(sample_coords)
        k_actual = min(k_neighbors + 1, len(sample_coords))
        distances_grid, _ = tree.query(grid_coords, k=k_actual)
        
        if distances_grid.ndim == 1:
            nn_dist_grid = distances_grid
        else:
            nn_dist_grid = distances_grid[:, -1]
        
        # Normalize using sample statistics
        if nn_dist_grid.std() > 1e-9:
            # Estimate mean/std from sample density
            tree_samples = KDTree(sample_coords)
            dists_samples, _ = tree_samples.query(sample_coords, k=k_actual)
            if dists_samples.ndim == 2:
                nn_samples = dists_samples[:, -1]
                nn_dist_grid = (nn_dist_grid - nn_samples.mean()) / nn_samples.std()
        
        features_list.append(nn_dist_grid.reshape(-1, 1))
    
    # 3. Depth proxy
    if include_depth:
        depth_grid = -grid_coords[:, 2]
        # Normalize using sample statistics
        depth_samples = -sample_coords[:, 2]
        if depth_samples.std() > 1e-9:
            depth_grid = (depth_grid - depth_samples.mean()) / depth_samples.std()
        features_list.append(depth_grid.reshape(-1, 1))
    
    # 4. Polynomial features
    if include_polynomial and 'poly_features' in scalers_dict:
        poly = scalers_dict['poly_features']
        poly_grid = poly.transform(grid_coords[:, :2])
        if poly_grid.shape[1] > 2:
            # Skip linear terms (first 2 columns)
            poly_features = poly_grid[:, 2:]
            features_list.append(poly_features)
    
    # Combine all features
    features = np.column_stack(features_list)
    
    return features

