"""
Inverse Distance Weighted (IDW) interpolation for ore grade estimation.

This module provides IDW interpolation functions for spatial estimation of
ore grades from drillhole sample data.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn for KDTree
try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available. IDW interpolation requires scikit-learn. "
        "Install with: pip install scikit-learn"
    )


def idw_interpolate(
    sample_coords: np.ndarray,
    sample_values: np.ndarray,
    query_coords: np.ndarray,
    k: int = 16,
    power: float = 2.0,
    eps: float = 1e-9
) -> np.ndarray:
    """
    Inverse Distance Weighted (IDW) interpolation.
    
    Estimates values at query locations using weighted average of k nearest
    neighbors, with weights inversely proportional to distance raised to power.
    
    This is commonly used as a baseline interpolation method for ore grade
    estimation in mining applications.
    
    Args:
        sample_coords: Sample coordinates (n_samples, 3) - X, Y, Z locations
        sample_values: Sample values (n_samples,) - e.g., ore grades
        query_coords: Query point coordinates (n_queries, 3) - points to estimate
        k: Number of nearest neighbors to use
        power: IDW exponent (typically 2.0). Higher values give more weight
               to closer points
        eps: Minimum distance to avoid division by zero
        
    Returns:
        Estimated values at query points (n_queries,)
        
    Example:
        >>> # Sample locations and grades
        >>> P = np.array([[100, 200, 50], [150, 250, 60], [120, 230, 55]])
        >>> V = np.array([2.5, 1.8, 2.1])  # Au grades (g/t)
        >>> # Query point
        >>> Q = np.array([[130, 220, 57]])
        >>> grade = idw_interpolate(P, V, Q, k=3, power=2.0)
        >>> print(f"Estimated grade: {grade[0]:.2f} g/t")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for IDW interpolation")
    
    sample_coords = np.asarray(sample_coords, dtype=np.float64)
    sample_values = np.asarray(sample_values, dtype=np.float64)
    query_coords = np.asarray(query_coords, dtype=np.float64)
    
    if sample_coords.ndim != 2 or query_coords.ndim != 2:
        raise ValueError("Coordinates must be 2D arrays (n_points, n_dim)")
    
    if sample_coords.shape[1] != query_coords.shape[1]:
        raise ValueError("Sample and query coordinates must have same dimensionality")
    
    if len(sample_values) != len(sample_coords):
        raise ValueError("Sample values must have same length as sample coordinates")
    
    if len(sample_coords) == 0:
        raise ValueError("Sample coordinates cannot be empty")
    
    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(sample_coords)
    
    # Find k nearest neighbors (or all neighbors if k > n_samples)
    k_actual = min(k, len(sample_coords))
    distances, indices = tree.query(query_coords, k=k_actual)
    
    # Handle case where query returns 1D arrays (single query point, k=1)
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)
    
    # Compute IDW weights: w = 1 / distance^power
    weights = 1.0 / np.maximum(distances, eps) ** power
    
    # Normalize weights to sum to 1 for each query point
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Weighted average: sum(weight * value) for each query point
    estimates = (sample_values[indices] * weights).sum(axis=1)
    
    return estimates


def compute_idw_residuals(
    coords: np.ndarray,
    values: np.ndarray,
    k: int = 16,
    power: float = 2.0,
    max_samples: int = 1000,
    leave_one_out: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IDW predictions and residuals using leave-one-out cross-validation.
    
    This is useful for hybrid IDW+ML models where ML learns the residuals
    between actual values and IDW predictions.
    
    Args:
        coords: Sample coordinates (n_samples, 3)
        values: Sample values (n_samples,)
        k: Number of nearest neighbors for IDW
        power: IDW exponent
        max_samples: Maximum number of samples to process (for speed).
                    If None, processes all samples.
        leave_one_out: If True, use true leave-one-out (slower but more accurate).
                      If False, use full IDW (faster but biased).
        
    Returns:
        Tuple of (idw_predictions, residuals):
            - idw_predictions: IDW predictions at sample locations (n_samples,)
            - residuals: Actual - Predicted (n_samples,)
            
    Example:
        >>> P = samples[['x', 'y', 'z']].values
        >>> V = samples['grade'].values
        >>> idw_pred, residuals = compute_idw_residuals(P, V, k=16)
        >>> print(f"Residual mean: {residuals.mean():.4f}")
    """
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    
    n_samples = len(coords)
    
    if max_samples and n_samples > max_samples:
        logger.warning(
            f"Processing {max_samples} of {n_samples} samples. "
            "Set max_samples=None for full processing."
        )
        n_process = max_samples
    else:
        n_process = n_samples
    
    idw_predictions = np.zeros(n_samples, dtype=np.float64)
    
    if leave_one_out:
        # True leave-one-out (more accurate but slower)
        for i in range(n_process):
            # Leave out sample i
            coords_train = np.delete(coords, i, axis=0)
            values_train = np.delete(values, i)
            
            # Predict at sample i
            query = coords[i:i+1]
            idw_predictions[i] = idw_interpolate(
                coords_train, values_train, query, k=k, power=power
            )[0]
        
        # For remaining samples, use full IDW (small bias but fast)
        if n_samples > n_process:
            idw_predictions[n_process:] = idw_interpolate(
                coords, values, coords[n_process:], k=k, power=power
            )
    else:
        # Use full IDW for all samples (faster but biased)
        idw_predictions = idw_interpolate(coords, values, coords, k=k, power=power)
    
    # Compute residuals
    residuals = values - idw_predictions
    
    logger.info(
        f"IDW residuals: mean={residuals.mean():.4f}, "
        f"std={residuals.std():.4f}, "
        f"range=[{residuals.min():.4f}, {residuals.max():.4f}]"
    )
    
    return idw_predictions, residuals

