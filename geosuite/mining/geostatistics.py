"""
Geostatistical modeling for orebody uncertainty quantification.

This module provides tools for:
- Variogram modeling and analysis
- Ordinary Kriging (OK) for grade estimation
- Sequential Gaussian Simulation (SGS) for uncertainty quantification
- Exceedance probability calculations
- Geochemical data processing and transformation
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning(
        "geopandas not available. Geospatial operations require geopandas. "
        "Install with: pip install geopandas"
    )

try:
    from skgstat import Variogram
    SKGSTAT_AVAILABLE = True
except ImportError:
    SKGSTAT_AVAILABLE = False
    logger.warning(
        "scikit-gstat not available. Variogram modeling requires scikit-gstat. "
        "Install with: pip install scikit-gstat"
    )

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    logger.warning(
        "pykrige not available. Kriging requires pykrige. "
        "Install with: pip install pykrige"
    )


def project_to_utm(
    gdf: gpd.GeoDataFrame,
    utm_zone: int = 32750,
    source_crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Project GeoDataFrame to UTM coordinates.
    
    Args:
        gdf: GeoDataFrame with WGS84 (lat/lon) coordinates
        utm_zone: UTM zone EPSG code (e.g., 32750 for UTM Zone 50S)
        source_crs: Source CRS (default: EPSG:4326 for WGS84)
        
    Returns:
        GeoDataFrame with UTM coordinates (x, y columns added)
        
    Example:
        >>> geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        >>> gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        >>> gdf = project_to_utm(gdf, utm_zone=32750)
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for coordinate projection")
    
    if gdf.crs is None:
        gdf = gdf.set_crs(source_crs)
    
    # Project to UTM
    gdf_utm = gdf.to_crs(f'EPSG:{utm_zone}')
    
    # Extract UTM coordinates
    gdf_utm['x'] = gdf_utm.geometry.x
    gdf_utm['y'] = gdf_utm.geometry.y
    
    return gdf_utm


def create_geochemical_dataframe(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    sample_ids: Optional[np.ndarray] = None,
    value_name: str = 'grade',
    sample_type: Optional[np.ndarray] = None
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame from geochemical sample data.
    
    Args:
        lons: Longitudes (degrees)
        lats: Latitudes (degrees)
        values: Sample values (e.g., grades)
        sample_ids: Optional sample IDs
        value_name: Name for value column (default: 'grade')
        sample_type: Optional sample type array
        
    Returns:
        GeoDataFrame with geometry and value columns
        
    Example:
        >>> gdf = create_geochemical_dataframe(lons, lats, gold_ppm, value_name='Au_ppm')
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required")
    
    n_samples = len(lons)
    
    if sample_ids is None:
        sample_ids = [f'S{i:06d}' for i in range(n_samples)]
    
    data = {
        'sample_id': sample_ids,
        'longitude': lons,
        'latitude': lats,
        value_name: values
    }
    
    if sample_type is not None:
        data['sample_type'] = sample_type
    
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
    
    return gdf


def compute_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    model: str = 'spherical',
    maxlag: float = 50000.0,
    n_lags: int = 20,
    normalize: bool = False
) -> Variogram:
    """
    Compute experimental variogram from sample data.
    
    Args:
        coords: Sample coordinates (n_samples, 2) - [X, Y]
        values: Sample values (n_samples,)
        model: Variogram model type ('spherical', 'exponential', 'gaussian')
        maxlag: Maximum lag distance (same units as coordinates)
        n_lags: Number of lag bins
        normalize: Whether to normalize variogram
        
    Returns:
        Variogram object with fitted model
        
    Example:
        >>> coords = gdf[['x', 'y']].values
        >>> values = gdf['log_Au'].values
        >>> V = compute_variogram(coords, values, model='spherical', maxlag=50000)
        >>> print(f"Range: {V.parameters[0]/1000:.1f} km")
    """
    if not SKGSTAT_AVAILABLE:
        raise ImportError("scikit-gstat is required for variogram modeling")
    
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Coordinates must be 2D array with at least 2 columns (X, Y)")
    
    if len(coords) != len(values):
        raise ValueError("Coordinates and values must have same length")
    
    # Extract X, Y (ignore Z if present)
    coords_2d = coords[:, :2]
    
    logger.info(f"Computing variogram for {len(coords):,} samples...")
    logger.info(f"  Model: {model}, maxlag: {maxlag:.1f}, n_lags: {n_lags}")
    
    V = Variogram(
        coords_2d,
        values,
        model=model,
        maxlag=maxlag,
        n_lags=n_lags,
        normalize=normalize
    )
    
    logger.info(f"✓ Variogram computed")
    logger.info(f"  Range: {V.parameters[0]:.1f} units")
    logger.info(f"  Sill: {V.parameters[1]:.4f}")
    logger.info(f"  Nugget: {V.parameters[2]:.4f}")
    
    return V


def ordinary_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    gridx: np.ndarray,
    gridy: np.ndarray,
    variogram_model: str = 'spherical',
    variogram_parameters: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    enable_plotting: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Ordinary Kriging (OK) interpolation.
    
    Args:
        coords: Sample coordinates (n_samples, 2) - [X, Y]
        values: Sample values (n_samples,)
        gridx: Grid X coordinates (1D array)
        gridy: Grid Y coordinates (1D array)
        variogram_model: Variogram model type ('spherical', 'exponential', 'gaussian')
        variogram_parameters: Variogram parameters dict with keys 'sill', 'range', 'nugget'.
                             If None, will be estimated from data.
        verbose: Whether to print progress
        enable_plotting: Whether to enable plotting in pykrige
        
    Returns:
        Tuple of (kriged_values, kriging_variance):
            - kriged_values: Kriged estimates on grid (ny, nx)
            - kriging_variance: Kriging variance (uncertainty) on grid (ny, nx)
            
    Example:
        >>> gridx = np.linspace(x_min, x_max, 100)
        >>> gridy = np.linspace(y_min, y_max, 100)
        >>> z_ok, ss_ok = ordinary_kriging(
        ...     coords, values, gridx, gridy,
        ...     variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187}
        ... )
    """
    if not PYKRIGE_AVAILABLE:
        raise ImportError("pykrige is required for kriging")
    
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    gridx = np.asarray(gridx, dtype=np.float64)
    gridy = np.asarray(gridy, dtype=np.float64)
    
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Coordinates must be 2D array with at least 2 columns")
    
    if len(coords) != len(values):
        raise ValueError("Coordinates and values must have same length")
    
    # Extract X, Y (ignore Z if present)
    coords_2d = coords[:, :2]
    
    # Default variogram parameters (spherical model)
    if variogram_parameters is None:
        variogram_parameters = {'sill': 1.0, 'range': 1000.0, 'nugget': 0.0}
    
    logger.info(f"Running Ordinary Kriging on {len(gridx)} × {len(gridy)} = {len(gridx) * len(gridy):,} grid points...")
    
    OK = OrdinaryKriging(
        coords_2d[:, 0],  # X coordinates
        coords_2d[:, 1],  # Y coordinates
        values,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        verbose=verbose,
        enable_plotting=enable_plotting
    )
    
    z, ss = OK.execute('grid', gridx, gridy)
    
    logger.info(f"✓ Kriging complete")
    logger.info(f"  Kriged range: [{z.min():.4f}, {z.max():.4f}]")
    logger.info(f"  Variance range: [{ss.min():.4f}, {ss.max():.4f}]")
    
    return z, ss


def sequential_gaussian_simulation(
    coords: np.ndarray,
    values: np.ndarray,
    gridx: np.ndarray,
    gridy: np.ndarray,
    n_realizations: int = 50,
    variogram_model: str = 'spherical',
    variogram_parameters: Optional[Dict[str, float]] = None,
    noise_level: float = 0.1,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Perform Sequential Gaussian Simulation (SGS) for uncertainty quantification.
    
    This is a simplified SGS implementation. For production use, consider
    using specialized libraries like PyGSLIB or SGeMS.
    
    Args:
        coords: Sample coordinates (n_samples, 2) - [X, Y]
        values: Sample values (n_samples,)
        gridx: Grid X coordinates (1D array)
        gridy: Grid Y coordinates (1D array)
        n_realizations: Number of realizations to generate
        variogram_model: Variogram model type
        variogram_parameters: Variogram parameters dict
        noise_level: Standard deviation of noise to add to sample data
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Simulation stack (n_realizations, ny, nx) - array of realizations
        
    Example:
        >>> sim_stack = sequential_gaussian_simulation(
        ...     coords, values, gridx, gridy,
        ...     n_realizations=50,
        ...     variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187}
        ... )
        >>> print(f"Generated {sim_stack.shape[0]} realizations")
    """
    if not PYKRIGE_AVAILABLE:
        raise ImportError("pykrige is required for simulation")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    
    if variogram_parameters is None:
        variogram_parameters = {'sill': 1.0, 'range': 1000.0, 'nugget': 0.0}
    
    simulations = []
    
    if verbose:
        logger.info(f"Generating {n_realizations} SGS realizations...")
    
    for i in range(n_realizations):
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Realization {i+1}/{n_realizations}...")
        
        # Perturb sample data with random noise (simplified SGS)
        # In production, use proper SGS algorithm (PyGSLIB, SGeMS, or custom)
        noise = np.random.normal(0, noise_level, len(coords))
        perturbed = values + noise
        
        # Krige with perturbed data
        z_sim, _ = ordinary_kriging(
            coords, perturbed, gridx, gridy,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            verbose=False,
            enable_plotting=False
        )
        
        simulations.append(z_sim)
    
    # Stack realizations
    sim_stack = np.stack(simulations)  # Shape: (n_realizations, ny, nx)
    
    if verbose:
        logger.info(f"✓ Generated {n_realizations} realizations")
        logger.info(f"  Simulation stack shape: {sim_stack.shape}")
    
    return sim_stack


def compute_simulation_statistics(
    sim_stack: np.ndarray,
    transform_back: Optional[callable] = None
) -> Dict[str, np.ndarray]:
    """
    Compute statistics across simulation realizations.
    
    Args:
        sim_stack: Simulation stack (n_realizations, ny, nx)
        transform_back: Optional function to transform values back (e.g., np.expm1 for log-transformed data)
        
    Returns:
        Dictionary with statistics:
            - 'mean': Mean across realizations
            - 'std': Standard deviation across realizations
            - 'p10': 10th percentile
            - 'p50': 50th percentile (median)
            - 'p90': 90th percentile
            
    Example:
        >>> stats = compute_simulation_statistics(sim_stack, transform_back=np.expm1)
        >>> mean_grade = stats['mean']
        >>> std_grade = stats['std']
    """
    sim_stack = np.asarray(sim_stack, dtype=np.float64)
    
    if sim_stack.ndim != 3:
        raise ValueError("Simulation stack must be 3D (n_realizations, ny, nx)")
    
    # Compute statistics
    mean_sim = np.mean(sim_stack, axis=0)
    std_sim = np.std(sim_stack, axis=0)
    p10_sim = np.percentile(sim_stack, 10, axis=0)
    p50_sim = np.percentile(sim_stack, 50, axis=0)
    p90_sim = np.percentile(sim_stack, 90, axis=0)
    
    # Transform back if needed
    if transform_back is not None:
        mean_sim = transform_back(mean_sim)
        std_sim = transform_back(std_sim)
        p10_sim = transform_back(p10_sim)
        p50_sim = transform_back(p50_sim)
        p90_sim = transform_back(p90_sim)
    
    return {
        'mean': mean_sim,
        'std': std_sim,
        'p10': p10_sim,
        'p50': p50_sim,
        'p90': p90_sim
    }


def compute_exceedance_probability(
    sim_stack: np.ndarray,
    cutoff: float,
    transform_cutoff: Optional[callable] = None
) -> np.ndarray:
    """
    Compute probability of exceeding a cutoff value.
    
    Args:
        sim_stack: Simulation stack (n_realizations, ny, nx) - in simulation space
        cutoff: Cutoff value (in original space, e.g., ppm)
        transform_cutoff: Optional function to transform cutoff to simulation space.
                         If sim_stack is in log space and cutoff is in original space,
                         use np.log1p to transform cutoff: transform_cutoff=np.log1p
        
    Returns:
        Exceedance probability array (ny, nx) - proportion of realizations exceeding cutoff
        
    Example:
        >>> # For log-transformed data in sim_stack
        >>> # If cutoff is in original space (ppm), transform it to log space
        >>> prob_exceed = compute_exceedance_probability(
        ...     sim_stack, cutoff=0.5, transform_cutoff=np.log1p
        ... )
        >>> print(f"Blocks with >50% prob: {(prob_exceed > 0.5).sum()}")
        
        >>> # If sim_stack is already in original space
        >>> prob_exceed = compute_exceedance_probability(sim_stack, cutoff=0.5)
    """
    sim_stack = np.asarray(sim_stack, dtype=np.float64)
    
    if transform_cutoff is not None:
        # Transform cutoff to simulation space
        # e.g., if sim_stack is in log space and cutoff is in original space:
        # cutoff_sim = log1p(cutoff)
        cutoff_sim = transform_cutoff(cutoff)
    else:
        # Cutoff is already in simulation space
        cutoff_sim = cutoff
    
    # Count realizations exceeding cutoff
    prob_exceed = (sim_stack > cutoff_sim).mean(axis=0)
    
    return prob_exceed


def log_transform(values: np.ndarray, add_one: bool = True) -> np.ndarray:
    """
    Log transform values (optionally adding 1 to avoid log(0)).
    
    Args:
        values: Values to transform
        add_one: If True, use log1p (log(1+x)); if False, use log(x)
        
    Returns:
        Log-transformed values
        
    Example:
        >>> log_grades = log_transform(grades, add_one=True)
    """
    values = np.asarray(values, dtype=np.float64)
    
    if add_one:
        return np.log1p(values)
    else:
        return np.log(np.maximum(values, 1e-10))  # Avoid log(0)


def exp_transform(log_values: np.ndarray, subtract_one: bool = True) -> np.ndarray:
    """
    Exponential transform values back from log space.
    
    Args:
        log_values: Log-transformed values
        subtract_one: If True, use expm1 (exp(x)-1); if False, use exp(x)
        
    Returns:
        Original-scale values
        
    Example:
        >>> grades = exp_transform(log_grades, subtract_one=True)
    """
    log_values = np.asarray(log_values, dtype=np.float64)
    
    if subtract_one:
        return np.expm1(log_values)
    else:
        return np.exp(log_values)

