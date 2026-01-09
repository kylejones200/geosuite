"""
Ore Grade Forecasting with Geochemistry and Machine Learning.

This module provides multi-method ore grade forecasting combining:
- Ordinary Kriging (geostatistical baseline)
- Gaussian Process Regression (probabilistic ML with uncertainty)
- XGBoost (gradient boosting for production forecasting)

All methods support spatial cross-validation to prevent data leakage.
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
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning(
        "geopandas not available. Geospatial operations require geopandas. "
        "Install with: pip install geopandas"
    )

try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available. Forecasting requires scikit-learn. "
        "Install with: pip install scikit-learn"
    )

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost not available. XGBoost forecasting requires xgboost. "
        "Install with: pip install xgboost"
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

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning(
        "scipy not available. Spatial operations require scipy. "
        "Install with: pip install scipy"
    )

from .geostatistics import (
    compute_variogram,
    ordinary_kriging,
    log_transform,
    exp_transform
)


def generate_synthetic_geochemical_data(
    n_samples: int = 250,
    region_bounds: Optional[Dict[str, float]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic geochemical data matching NGSA structure.
    
    For demonstration, generates synthetic data. In production, download from:
    https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/122101
    
    Args:
        n_samples: Number of samples to generate
        region_bounds: Optional dict with 'lon_min', 'lon_max', 'lat_min', 'lat_max'
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sample locations, element concentrations, and lithology
        
    Example:
        >>> df = generate_synthetic_geochemical_data(n_samples=250, random_state=42)
        >>> print(f"Generated {len(df)} samples with Au range: {df['Au'].min():.3f} - {df['Au'].max():.3f} ppm")
    """
    if region_bounds is None:
        region_bounds = {'lon_min': 118.0, 'lon_max': 123.0, 'lat_min': -32.0, 'lat_max': -28.0}
    
    np.random.seed(random_state)
    
    lon = np.random.uniform(region_bounds['lon_min'], region_bounds['lon_max'], n_samples)
    lat = np.random.uniform(region_bounds['lat_min'], region_bounds['lat_max'], n_samples)
    
    # Normalize coordinates
    x_norm = (lon - lon.min()) / (lon.max() - lon.min())
    y_norm = (lat - lat.min()) / (lat.max() - lat.min())
    
    # Create mineralization zones
    zone1 = np.exp(-((x_norm - 0.3) ** 2 + (y_norm - 0.4) ** 2) / 0.01)
    zone2 = np.exp(-((x_norm - 0.7) ** 2 + (y_norm - 0.6) ** 2) / 0.015)
    zone3 = np.exp(-((x_norm - 0.5) ** 2 + (y_norm - 0.2) ** 2) / 0.008)
    mineralization = zone1 + zone2 + zone3
    
    # Generate gold grades with spatial structure
    log_au_base = mineralization * 3.0 + np.random.randn(n_samples) * 0.5
    au_ppm = np.exp(log_au_base) * 0.01
    au_ppm = np.clip(au_ppm, 0.001, 5.0)
    
    # Generate correlated geochemical elements
    cu_ppm = au_ppm * 50 + np.random.randn(n_samples) * 10
    as_ppm = au_ppm * 30 + np.random.randn(n_samples) * 5
    pb_ppm = au_ppm * 20 + np.random.randn(n_samples) * 8
    s_pct = au_ppm * 0.3 + np.random.randn(n_samples) * 0.1
    fe_pct = 4.0 + mineralization * 2.0 + np.random.randn(n_samples) * 1.0
    
    # Generate lithology with spatial correlation
    lithology_types = ['granite', 'basalt', 'sediment', 'greenstone']
    lithology_probs = mineralization / mineralization.sum()
    lithology_probs = np.column_stack([
        lithology_probs * 0.2,
        lithology_probs * 0.3,
        (1 - lithology_probs) * 0.3,
        lithology_probs * 0.4
    ])
    lithology_probs = lithology_probs / lithology_probs.sum(axis=1, keepdims=True)
    lithology = np.array([
        np.random.choice(lithology_types, p=probs) for probs in lithology_probs
    ])
    
    df = pd.DataFrame({
        'longitude': lon,
        'latitude': lat,
        'Au': au_ppm,
        'Cu': cu_ppm,
        'As': as_ppm,
        'Pb': pb_ppm,
        'S': s_pct,
        'Fe': fe_pct,
        'lithology': lithology,
        'sample_id': [f'NGSA_{i:04d}' for i in range(n_samples)]
    })
    
    logger.info(f"Generated {len(df)} synthetic geochemical samples")
    logger.info(f"  Au range: {au_ppm.min():.3f} - {au_ppm.max():.3f} ppm")
    logger.info(f"  Au mean: {au_ppm.mean():.3f} ppm")
    
    return df


def prepare_spatial_features(
    df: pd.DataFrame,
    target_crs: str = 'EPSG:32750'
) -> gpd.GeoDataFrame:
    """
    Convert to projected CRS and extract spatial features.
    
    Args:
        df: DataFrame with longitude, latitude, Au, and covariates
        target_crs: UTM zone for Western Australia (zone 50S by default)
        
    Returns:
        GeoDataFrame with x, y (km), log_Au, and features
        
    Example:
        >>> gdf = prepare_spatial_features(df)
        >>> print(f"Prepared {len(gdf)} samples in UTM coordinates")
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for coordinate projection")
    
    df = df[df['Au'] > 0].copy()
    df['log_Au'] = log_transform(df['Au'].values, add_one=True)
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    gdf = gdf.to_crs(target_crs)
    
    # Convert to kilometers
    gdf['x'] = gdf.geometry.x / 1000
    gdf['y'] = gdf.geometry.y / 1000
    
    logger.info(f'Prepared {len(gdf)} samples')
    logger.info(f"  Au range: {gdf['Au'].min():.3f} - {gdf['Au'].max():.3f} ppm")
    logger.info(f"  Mean Au: {gdf['Au'].mean():.3f} ppm")
    logger.info(
        f"  Spatial extent: {gdf['x'].max() - gdf['x'].min():.1f} km × "
        f"{gdf['y'].max() - gdf['y'].min():.1f} km"
    )
    
    return gdf


def create_spatial_folds(gdf: gpd.GeoDataFrame, n_folds: int = 5) -> np.ndarray:
    """
    Create spatial cross-validation folds to prevent leakage.
    
    Uses x-coordinate bands to ensure train/test spatial separation.
    
    Args:
        gdf: GeoDataFrame with sample locations
        n_folds: Number of folds
        
    Returns:
        Array of fold assignments (0 to n_folds-1)
        
    Example:
        >>> groups = create_spatial_folds(gdf, n_folds=5)
        >>> print(f"Created {len(np.unique(groups))} spatial folds")
    """
    groups = pd.qcut(gdf['x'], n_folds, labels=False, duplicates='drop')
    
    logger.info(f'\nCreated {n_folds} spatial folds:')
    for fold in range(n_folds):
        n = (groups == fold).sum()
        logger.info(f'  Fold {fold}: {n} samples')
    
    return groups.values


def fit_variogram_for_forecasting(
    gdf: gpd.GeoDataFrame,
    maxlag: Union[str, float] = 'median',
    n_lags: int = 25
) -> Variogram:
    """
    Fit experimental and theoretical variogram for spatial correlation.
    
    Args:
        gdf: GeoDataFrame with x, y (km) and log_Au
        maxlag: Maximum lag distance ('median', 'mean', or float in same units as coordinates)
        n_lags: Number of lag bins
        
    Returns:
        Variogram model fitted to data (range in same units as coordinates - km)
        
    Example:
        >>> V = fit_variogram_for_forecasting(gdf)
        >>> print(f"Variogram range: {V.range:.1f} km, sill: {V.sill:.3f}")
    """
    if not SKGSTAT_AVAILABLE:
        raise ImportError("scikit-gstat is required for variogram modeling")
    
    coords = np.column_stack([gdf['x'].values, gdf['y'].values])
    values = gdf['log_Au'].values
    
    V = Variogram(
        coords,
        values,
        model='spherical',
        maxlag=maxlag,
        n_lags=n_lags
    )
    
    logger.info('\nVariogram Parameters:')
    logger.info(f'  Model: {V.model.__name__}')
    logger.info(f'  Sill: {V.sill:.3f}')
    logger.info(f'  Range: {V.range:.1f} km')
    logger.info(f'  Nugget: {V.nugget:.3f}')
    logger.info(f'  Nugget/Sill ratio: {V.nugget / V.sill:.2%}')
    
    return V


def ordinary_kriging_predict(
    gdf: gpd.GeoDataFrame,
    grid_resolution: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Ordinary Kriging on a regular grid.
    
    Args:
        gdf: GeoDataFrame with x, y (km) and log_Au
        grid_resolution: Grid resolution (n × n)
        
    Returns:
        Tuple of (grid_x, grid_y, predictions_ppm, variance):
            - grid_x: 1D X coordinates (km)
            - grid_y: 1D Y coordinates (km)
            - predictions_ppm: Kriged estimates in ppm (ny, nx)
            - variance: Kriging variance (ny, nx)
            
    Example:
        >>> gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=100)
        >>> print(f"Kriging predictions: {ok_ppm.min():.3f} - {ok_ppm.max():.3f} ppm")
    """
    if not PYKRIGE_AVAILABLE:
        raise ImportError("pykrige is required for kriging")
    
    gx = np.linspace(gdf['x'].min(), gdf['x'].max(), grid_resolution)
    gy = np.linspace(gdf['y'].min(), gdf['y'].max(), grid_resolution)
    
    # Use variogram from geostatistics module
    coords = np.column_stack([gdf['x'].values, gdf['y'].values])
    values = gdf['log_Au'].values
    
    # Fit variogram to get parameters
    # Note: Coordinates are in km, so maxlag should be in km too
    # Use 'median' as maxlag to automatically determine appropriate lag distance
    V = fit_variogram_for_forecasting(gdf, maxlag='median', n_lags=25)
    variogram_params = {
        'sill': V.parameters[1],
        'range': V.parameters[0],  # Already in km
        'nugget': V.parameters[2]
    }
    
    # Ordinary kriging (coordinates in km, variogram parameters in km)
    z, ss = ordinary_kriging(
        coords,
        values,
        gx,
        gy,
        variogram_model='spherical',
        variogram_parameters=variogram_params,
        verbose=False,
        enable_plotting=False
    )
    
    # Back-transform to ppm
    z_ppm = exp_transform(z, subtract_one=True)
    
    logger.info('\nOrdinary Kriging Results:')
    logger.info(f'  Grid size: {grid_resolution} × {grid_resolution}')
    logger.info(f'  Predicted Au range: {z_ppm.min():.3f} - {z_ppm.max():.3f} ppm')
    logger.info(f'  Mean kriging variance: {ss.mean():.3f}')
    
    return gx, gy, z_ppm, ss


def train_gaussian_process(
    gdf: gpd.GeoDataFrame,
    groups: np.ndarray,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    n_folds: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train Gaussian Process Regressor with spatial cross-validation.
    
    Args:
        gdf: GeoDataFrame with features and log_Au
        groups: Spatial fold assignments for cross-validation
        numeric_features: List of numeric feature columns
        categorical_features: List of categorical feature columns
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, predictions, uncertainties, metrics):
            - trained_model: Fitted sklearn Pipeline
            - predictions: CV predictions (log space)
            - uncertainties: Prediction standard deviations
            - metrics: Dict with 'mae', 'rmse', 'coverage'
            
    Example:
        >>> gp_model, gp_pred, gp_std, metrics = train_gaussian_process(gdf, groups)
        >>> print(f"GPR MAE: {metrics['mae']:.3f}, Coverage: {metrics['coverage']:.1%}")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for Gaussian Process Regression")
    
    if numeric_features is None:
        numeric_features = ['x', 'y', 'Cu', 'As', 'Fe', 'S', 'Pb']
    if categorical_features is None:
        categorical_features = ['lithology']
    
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        ), categorical_features)
    ])
    
    # Define GPR kernel
    kernel = (
        ConstantKernel(1.0, (0.001, 1000.0)) *
        Matern(length_scale=1.0, length_scale_bounds=(0.01, 100.0), nu=1.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
    )
    
    gp_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('gpr', GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=random_state
        ))
    ])
    
    X = gdf[numeric_features + categorical_features]
    y = gdf['log_Au'].values
    
    pred_mu = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    
    logger.info('\nGaussian Process Cross-Validation:')
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        gp_pipeline.fit(X_train, y_train)
        
        X_test_transformed = gp_pipeline.named_steps['preprocessor'].transform(X_test)
        mu, std = gp_pipeline.named_steps['gpr'].predict(X_test_transformed, return_std=True)
        
        pred_mu[test_idx] = mu
        pred_std[test_idx] = std
        
        fold_mae = mean_absolute_error(y_test, mu)
        fold_rmse = np.sqrt(mean_squared_error(y_test, mu))
        logger.info(f'  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}')
    
    mae = mean_absolute_error(y, pred_mu)
    rmse = np.sqrt(mean_squared_error(y, pred_mu))
    
    # Compute calibration (95% confidence coverage)
    z_scores = np.abs(y - pred_mu) / np.maximum(pred_std, 1e-6)
    coverage_95 = (z_scores < 1.96).mean()
    
    logger.info(f'\nGPR Overall Performance:')
    logger.info(f'  MAE: {mae:.3f} log(ppm)')
    logger.info(f'  RMSE: {rmse:.3f} log(ppm)')
    logger.info(f'  95% Confidence Coverage: {coverage_95:.1%}')
    logger.info(f'  Mean Prediction Std: {pred_std.mean():.3f}')
    
    # Train final model on all data
    gp_pipeline.fit(X, y)
    
    return gp_pipeline, pred_mu, pred_std, {
        'mae': mae,
        'rmse': rmse,
        'coverage': coverage_95
    }


def train_xgboost(
    gdf: gpd.GeoDataFrame,
    groups: np.ndarray,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    n_folds: int = 5,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42
) -> Tuple[Pipeline, np.ndarray, Dict[str, float]]:
    """
    Train XGBoost regressor with spatial cross-validation.
    
    Args:
        gdf: GeoDataFrame with features and log_Au
        groups: Spatial fold assignments
        numeric_features: List of numeric feature columns
        categorical_features: List of categorical feature columns
        n_folds: Number of CV folds
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, predictions, metrics):
            - trained_model: Fitted sklearn Pipeline
            - predictions: CV predictions (log space)
            - metrics: Dict with 'mae', 'rmse'
            
    Example:
        >>> xgb_model, xgb_pred, metrics = train_xgboost(gdf, groups)
        >>> print(f"XGBoost MAE: {metrics['mae']:.3f}")
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is required for XGBoost forecasting")
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for preprocessing")
    
    if numeric_features is None:
        numeric_features = ['x', 'y', 'Cu', 'As', 'Fe', 'S', 'Pb']
    if categorical_features is None:
        categorical_features = ['lithology']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        ), categorical_features)
    ])
    
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    X = gdf[numeric_features + categorical_features]
    y = gdf['log_Au'].values
    
    pred = np.zeros_like(y)
    
    logger.info('\nXGBoost Cross-Validation:')
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        xgb_pipeline.fit(X_train, y_train)
        pred[test_idx] = xgb_pipeline.predict(X_test)
        
        fold_mae = mean_absolute_error(y_test, pred[test_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_test, pred[test_idx]))
        logger.info(f'  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}')
    
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    
    logger.info(f'\nXGBoost Overall Performance:')
    logger.info(f'  MAE: {mae:.3f} log(ppm)')
    logger.info(f'  RMSE: {rmse:.3f} log(ppm)')
    
    # Train final model on all data
    xgb_pipeline.fit(X, y)
    
    # Feature importance
    feature_names = numeric_features + list(
        xgb_pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_features)
    )
    importances = xgb_pipeline.named_steps['xgb'].feature_importances_
    
    logger.info('\nTop Feature Importances:')
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f'  {name}: {imp:.3f}')
    
    return xgb_pipeline, pred, {'mae': mae, 'rmse': rmse}


def create_prediction_grid(
    gdf: gpd.GeoDataFrame,
    gp_model: Pipeline,
    xgb_model: Pipeline,
    resolution: int = 150,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Generate grade predictions on a regular grid for all three methods.
    
    Args:
        gdf: GeoDataFrame with sample locations and features
        gp_model: Trained Gaussian Process model
        xgb_model: Trained XGBoost model
        resolution: Grid resolution (n × n)
        numeric_features: List of numeric feature columns
        categorical_features: List of categorical feature columns
        
    Returns:
        Dictionary with grid predictions:
            - 'grid_x': 2D X coordinates (km)
            - 'grid_y': 2D Y coordinates (km)
            - 'gp_mean': GPR predictions (ppm)
            - 'gp_std': GPR uncertainties (std dev)
            - 'xgb_pred': XGBoost predictions (ppm)
            - 'ok_mean': Ordinary Kriging predictions (ppm)
            - 'ok_var': Ordinary Kriging variances
            
    Example:
        >>> grid_results = create_prediction_grid(gdf, gp_model, xgb_model)
        >>> print(f"Grid shape: {grid_results['gp_mean'].shape}")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for spatial operations")
    
    if numeric_features is None:
        numeric_features = ['x', 'y', 'Cu', 'As', 'Fe', 'S', 'Pb']
    if categorical_features is None:
        categorical_features = ['lithology']
    
    gx = np.linspace(gdf['x'].min(), gdf['x'].max(), resolution)
    gy = np.linspace(gdf['y'].min(), gdf['y'].max(), resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)
    
    # Find nearest samples for feature values
    tree = cKDTree(np.column_stack([gdf['x'], gdf['y']]))
    _, nearest_idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    
    grid_features = pd.DataFrame({
        'x': grid_x.ravel(),
        'y': grid_y.ravel(),
        'Cu': gdf.iloc[nearest_idx]['Cu'].values,
        'As': gdf.iloc[nearest_idx]['As'].values,
        'Fe': gdf.iloc[nearest_idx]['Fe'].values,
        'S': gdf.iloc[nearest_idx]['S'].values,
        'Pb': gdf.iloc[nearest_idx]['Pb'].values,
        'lithology': gdf.iloc[nearest_idx]['lithology'].values
    })
    
    # GPR predictions
    gp_transformed = gp_model.named_steps['preprocessor'].transform(grid_features)
    gp_mu, gp_std = gp_model.named_steps['gpr'].predict(gp_transformed, return_std=True)
    gp_ppm = exp_transform(gp_mu, subtract_one=True).reshape(grid_x.shape)
    gp_std_grid = gp_std.reshape(grid_x.shape)
    
    # XGBoost predictions
    xgb_pred = xgb_model.predict(grid_features)
    xgb_ppm = exp_transform(xgb_pred, subtract_one=True).reshape(grid_x.shape)
    
    # Ordinary Kriging
    gx_1d, gy_1d, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=resolution)
    
    logger.info(f'\nGrid Predictions Complete:')
    logger.info(f'  GPR Au range: {gp_ppm.min():.3f} - {gp_ppm.max():.3f} ppm')
    logger.info(f'  XGB Au range: {xgb_ppm.min():.3f} - {xgb_ppm.max():.3f} ppm')
    logger.info(f'  OK Au range: {ok_ppm.min():.3f} - {ok_ppm.max():.3f} ppm')
    
    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'gp_mean': gp_ppm,
        'gp_std': gp_std_grid,
        'xgb_pred': xgb_ppm,
        'ok_mean': ok_ppm,
        'ok_var': ok_var
    }


def analyze_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Analyze uncertainty calibration by binning predictions by confidence.
    
    Well-calibrated models show actual RMSE matching predicted uncertainty.
    
    Args:
        y_true: True values (log space)
        y_pred: Predicted values (log space)
        y_std: Predicted standard deviations
        n_bins: Number of bins for calibration analysis
        
    Returns:
        DataFrame with calibration statistics:
            - predicted_std: Mean predicted std in bin
            - actual_rmse: Actual RMSE in bin
            - n_samples: Number of samples in bin
            
    Example:
        >>> calib_df = analyze_uncertainty_calibration(y_true, y_pred, y_std)
        >>> print(f"Calibration correlation: {calib_df[['predicted_std', 'actual_rmse']].corr().iloc[0,1]:.3f}")
    """
    bins = pd.qcut(y_std, n_bins, duplicates='drop')
    calibration_data = []
    
    for bin_label in bins.cat.categories:
        mask = bins == bin_label
        bin_std = y_std[mask].mean()
        bin_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        calibration_data.append({
            'predicted_std': bin_std,
            'actual_rmse': bin_rmse,
            'n_samples': mask.sum()
        })
    
    calib_df = pd.DataFrame(calibration_data)
    
    logger.info('\nUncertainty Calibration:')
    logger.info(calib_df.to_string(index=False))
    
    correlation = np.corrcoef(calib_df['predicted_std'], calib_df['actual_rmse'])[0, 1]
    logger.info(f'\nCalibration Correlation: {correlation:.3f}')
    
    return calib_df


def compare_forecasting_methods(
    gpr_metrics: Dict[str, float],
    xgb_metrics: Dict[str, float],
    ok_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Comparative summary of all forecasting methods.
    
    Args:
        gpr_metrics: GPR metrics dict with 'mae', 'rmse', 'coverage'
        xgb_metrics: XGBoost metrics dict with 'mae', 'rmse'
        ok_metrics: Optional OK metrics (not used, kept for API compatibility)
        
    Example:
        >>> compare_forecasting_methods(gpr_metrics, xgb_metrics)
    """
    logger.info('\n' + '=' * 70)
    logger.info('MODEL COMPARISON SUMMARY')
    logger.info('=' * 70)
    
    logger.info('\nAccuracy Metrics:')
    logger.info('  Ordinary Kriging:    MAE = N/A (no CV), RMSE = N/A')
    logger.info(f"  Gaussian Process:    MAE = {gpr_metrics['mae']:.3f}, RMSE = {gpr_metrics['rmse']:.3f}")
    logger.info(f"  XGBoost:             MAE = {xgb_metrics['mae']:.3f}, RMSE = {xgb_metrics['rmse']:.3f}")
    
    logger.info('\nUncertainty Quantification:')
    logger.info('  Ordinary Kriging:    Kriging variance (but often overconfident)')
    logger.info(f"  Gaussian Process:    95% Coverage = {gpr_metrics['coverage']:.1%} (well-calibrated)")
    logger.info('  XGBoost:             None (point estimates only)')
    
    logger.info('\nComputational Efficiency:')
    logger.info('  Ordinary Kriging:    O(n³) - slow for large datasets')
    logger.info('  Gaussian Process:    O(n³) - same limitations')
    logger.info('  XGBoost:             O(n log n) - scales to millions of points')
    
    logger.info('\nBest Use Cases:')
    logger.info('  Ordinary Kriging:    Traditional geostatistics, spatial-only data')
    logger.info('  Gaussian Process:    When you need calibrated uncertainty + covariates')
    logger.info('  XGBoost:             Production forecasting with tight deadlines')
    logger.info('=' * 70)

