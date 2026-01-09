"""
Gaussian Process Regression (GPR) for reservoir property modeling.

This module provides GPR-based spatial modeling of reservoir properties
from sparse well log data. Supports cross-validation, uncertainty quantification,
and export to GRDECL format for reservoir simulators.

References:
    - Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes for machine learning.
      MIT Press.
    - SPE9 Reservoir Model: Industry-standard test case for reservoir simulation
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available. GPR modeling requires scikit-learn. "
        "Install with: pip install scikit-learn"
    )

# Try to import GPyTorch for GPU acceleration (optional)
try:
    import torch
    import gpytorch
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    logger.info("GPyTorch not available. GPU acceleration not available.")


class GPRReservoirModel:
    """
    Gaussian Process Regression model for reservoir property prediction.
    
    This class provides a convenient interface for training GPR models on
    sparse well log data and predicting reservoir properties on a full 3D grid
    with uncertainty quantification.
    
    Attributes:
        model: Trained GaussianProcessRegressor instance
        x_scaler: StandardScaler for input features
        y_scaler: StandardScaler for output values
        kernel: Kernel function used for GPR
        cv_scores: Cross-validation scores (if performed)
        cv_rmse: Cross-validation RMSE values (if performed)
    """
    
    def __init__(
        self,
        kernel: Optional[Any] = None,
        alpha: float = 0.1,
        n_restarts: int = 3,
        random_state: Optional[int] = 42
    ):
        """
        Initialize GPR reservoir model.
        
        Args:
            kernel: sklearn kernel instance. If None, uses default RBF + Matern
            alpha: Added diagonal noise variance (nugget)
            n_restarts: Number of optimizer restarts
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for GPR modeling")
        
        if kernel is None:
            # Default kernel: RBF + Matern
            kernel = (
                RBF(length_scale=[0.5, 0.5, 0.5], length_scale_bounds=(1e-3, 1e1)) +
                Matern(length_scale=0.5, nu=1.5)
            )
        
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.random_state = random_state
        
        self.model: Optional[GaussianProcessRegressor] = None
        self.x_scaler: Optional[StandardScaler] = None
        self.y_scaler: Optional[StandardScaler] = None
        self.cv_scores: Optional[List[float]] = None
        self.cv_rmse: Optional[List[float]] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        log_transform: bool = True
    ) -> 'GPRReservoirModel':
        """
        Fit GPR model to training data.
        
        Args:
            X: Training coordinates (n_samples, n_features) - typically (x, y, z)
            y: Training property values (n_samples,) - e.g., permeability
            log_transform: If True, apply log1p transform to y before fitting
            
        Returns:
            Self for method chaining
            
        Example:
            >>> X_train = np.random.rand(100, 3)  # (x, y, z) coordinates
            >>> y_train = np.random.lognormal(2, 1, 100)  # Permeability values
            >>> model = GPRReservoirModel()
            >>> model.fit(X_train, y_train, log_transform=True)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        if len(X) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Apply log transform if requested
        if log_transform:
            y = np.log1p(y)
        
        # Scale features and targets
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create and fit GPR model
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled, y_scaled)
        
        logger.info(f"GPR model fitted on {len(X)} samples")
        logger.info(f"Kernel: {self.model.kernel_}")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
        log_transform: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict reservoir property on new coordinates.
        
        Args:
            X: Prediction coordinates (n_samples, n_features)
            return_std: If True, return prediction uncertainty (std dev)
            log_transform: If True, apply inverse log1p transform to predictions
            
        Returns:
            Tuple of (predictions, std_dev):
                - predictions: Predicted property values
                - std_dev: Prediction uncertainty (None if return_std=False)
                
        Example:
            >>> X_pred = np.random.rand(1000, 3)  # Grid coordinates
            >>> pred, sigma = model.predict(X_pred, return_std=True)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.x_scaler is None or self.y_scaler is None:
            raise ValueError("Scalers not initialized. Model may not be fitted.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        
        # Scale input
        X_scaled = self.x_scaler.transform(X)
        
        # Predict
        if return_std:
            pred_scaled, sigma_scaled = self.model.predict(X_scaled, return_std=True)
        else:
            pred_scaled = self.model.predict(X_scaled)
            sigma_scaled = None
        
        # Inverse transform
        pred = self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        # Inverse log transform if applied
        if log_transform:
            pred = np.expm1(pred)
            # Note: sigma is not transformed back (it's in log space)
            # For uncertainty in original space, use: sigma_orig ≈ pred * sigma_log
        
        if return_std:
            return pred, sigma_scaled
        else:
            return pred, None
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        log_transform: bool = True,
        shuffle: bool = True
    ) -> Dict[str, Any]:
        """
        Perform K-fold cross-validation.
        
        Args:
            X: Training coordinates (n_samples, n_features)
            y: Training property values (n_samples,)
            n_splits: Number of CV folds
            log_transform: If True, apply log1p transform
            shuffle: If True, shuffle data before splitting
            
        Returns:
            Dictionary with CV results:
                - 'r2_mean': Mean R² score
                - 'r2_std': Std dev of R² scores
                - 'rmse_mean': Mean RMSE
                - 'rmse_std': Std dev of RMSE
                - 'r2_scores': List of R² scores per fold
                - 'rmse_scores': List of RMSE scores per fold
                
        Example:
            >>> cv_results = model.cross_validate(X_train, y_train, n_splits=5)
            >>> print(f"CV R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        # Apply log transform if requested
        if log_transform:
            y_log = np.log1p(y)
        else:
            y_log = y
        
        # Scale
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
        
        # Cross-validation
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        
        r2_scores = []
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_cv_train = X_scaled[train_idx]
            X_cv_val = X_scaled[val_idx]
            y_cv_train = y_scaled[train_idx]
            y_cv_val = y_scaled[val_idx]
            
            # Fit model on fold
            gpr_cv = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.alpha,
                random_state=self.random_state
            )
            gpr_cv.fit(X_cv_train, y_cv_train)
            
            # Predict on validation fold
            y_cv_pred = gpr_cv.predict(X_cv_val)
            
            # Convert back to original scale
            y_val_orig = y_scaler.inverse_transform(y_cv_val.reshape(-1, 1)).flatten()
            y_pred_orig = y_scaler.inverse_transform(y_cv_pred.reshape(-1, 1)).flatten()
            
            # Inverse log transform
            if log_transform:
                y_val_orig = np.expm1(y_val_orig)
                y_pred_orig = np.expm1(y_pred_orig)
            
            # Compute metrics
            r2 = r2_score(y_val_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
        
        # Store results
        self.cv_scores = r2_scores
        self.cv_rmse = rmse_scores
        
        return {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores
        }


def export_to_grdecl(
    property_3d: np.ndarray,
    property_name: str,
    filename: Union[str, Path],
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    nz: Optional[int] = None
) -> None:
    """
    Export 3D property array to GRDECL format.
    
    GRDECL format is used by reservoir simulators like Eclipse and INTERSECT.
    This function exports a 3D property array in Fortran column-major order.
    
    Args:
        property_3d: 3D property array (nx, ny, nz) or flattened array
        property_name: Property name (e.g., 'PERMX', 'PORO')
        filename: Output file path
        nx, ny, nz: Grid dimensions (required if property_3d is flattened)
        
    Example:
        >>> permx_3d = np.random.rand(24, 25, 15)  # 3D permeability array
        >>> export_to_grdecl(permx_3d, 'PERMX', 'permx.grdecl')
    """
    prop = np.asarray(property_3d, dtype=np.float64)
    
    # Reshape if flattened
    if prop.ndim == 1:
        if nx is None or ny is None or nz is None:
            raise ValueError("nx, ny, nz required if property_3d is 1D")
        prop = prop.reshape((nx, ny, nz), order='F')
    
    if prop.ndim != 3:
        raise ValueError("property_3d must be 3D array or 1D array with nx, ny, nz")
    
    # Flatten in Fortran order (column-major) for GRDECL
    values = prop.ravel(order='F')
    
    filename = Path(filename)
    
    # Write GRDECL file
    with open(filename, 'w') as f:
        f.write(f"{property_name}\n")
        
        # Write values in rows of 5
        for i in range(0, len(values), 5):
            row_values = values[i:i + 5]
            f.write("  ".join([f"{val:12.5E}" for val in row_values]) + "\n")
        
        f.write("/\n")
    
    logger.info(f"Exported {property_name} to {filename}")


def predict_on_grid(
    model: GPRReservoirModel,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    return_std: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict reservoir property on a regular 3D grid.
    
    Convenience function to generate predictions on a regular grid from
    a fitted GPR model.
    
    Args:
        model: Fitted GPRReservoirModel instance
        x_coords: X coordinate array (nx,)
        y_coords: Y coordinate array (ny,)
        z_coords: Z coordinate array (nz,)
        return_std: If True, return prediction uncertainty
        
    Returns:
        Tuple of (pred_3d, sigma_3d):
            - pred_3d: 3D prediction array (nx, ny, nz)
            - sigma_3d: 3D uncertainty array (nx, ny, nz) or None
            
    Example:
        >>> x = np.linspace(0, 1, 24)
        >>> y = np.linspace(0, 1, 25)
        >>> z = np.linspace(0, 1, 15)
        >>> pred, sigma = predict_on_grid(model, x, y, z)
    """
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Flatten to coordinate matrix
    coords = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])
    
    # Predict
    pred, sigma = model.predict(coords, return_std=return_std)
    
    # Reshape to 3D
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    pred_3d = pred.reshape((nx, ny, nz), order='F')
    
    if sigma is not None:
        sigma_3d = sigma.reshape((nx, ny, nz), order='F')
    else:
        sigma_3d = None
    
    return pred_3d, sigma_3d

