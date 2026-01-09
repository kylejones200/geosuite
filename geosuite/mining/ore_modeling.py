"""
Hybrid IDW + Machine Learning ore grade estimation.

This module provides a hybrid approach combining Inverse Distance Weighted (IDW)
interpolation with machine learning for improved ore grade estimation.
The ML model learns residuals from IDW baseline, capturing complex patterns
that IDW alone cannot model.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available. Ore modeling requires scikit-learn. "
        "Install with: pip install scikit-learn"
    )

from .interpolation import idw_interpolate, compute_idw_residuals
from .features import build_spatial_features, build_block_model_features


@dataclass
class HybridModelResults:
    """Results from hybrid IDW+ML model training."""
    idw_predictions: np.ndarray
    residuals: np.ndarray
    ml_predictions: np.ndarray
    final_predictions: np.ndarray
    cv_scores: Dict[str, List[float]]
    feature_importance: Optional[np.ndarray] = None


class HybridOreModel:
    """
    Hybrid IDW + Machine Learning model for ore grade estimation.
    
    This class combines IDW interpolation (baseline) with machine learning
    (residual learning) for improved ore grade estimation. The approach:
    1. Computes IDW baseline predictions
    2. Trains ML model on residuals (actual - IDW)
    3. Final prediction = IDW + ML residual
    
    This hybrid approach preserves spatial continuity from IDW while capturing
    complex patterns through ML.
    
    Attributes:
        idw_k: Number of neighbors for IDW
        idw_power: IDW exponent
        ml_model: Trained ML model (if fitted)
        feature_scalers: Dictionary of fitted feature scalers
        cv_scores: Cross-validation scores (if performed)
    """
    
    def __init__(
        self,
        idw_k: int = 16,
        idw_power: float = 2.0,
        ml_model_type: str = 'gradient_boosting',
        ml_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = 42
    ):
        """
        Initialize hybrid ore model.
        
        Args:
            idw_k: Number of nearest neighbors for IDW interpolation
            idw_power: IDW exponent (typically 2.0)
            ml_model_type: ML model type ('gradient_boosting' or 'random_forest')
            ml_params: Optional dictionary of ML model parameters
            random_state: Random seed for reproducibility
            
        Example:
            >>> model = HybridOreModel(
            ...     idw_k=16,
            ...     idw_power=2.0,
            ...     ml_model_type='gradient_boosting'
            ... )
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ore modeling")
        
        self.idw_k = idw_k
        self.idw_power = idw_power
        self.ml_model_type = ml_model_type
        self.random_state = random_state
        
        # Default ML parameters
        default_params = {
            'gradient_boosting': {
                'n_estimators': 200,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': random_state
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': random_state
            }
        }
        
        if ml_params is None:
            ml_params = default_params.get(ml_model_type, {})
        
        self.ml_params = ml_params
        
        # Will be set during fitting
        self.ml_model = None
        self.feature_scalers = None
        self.cv_scores = None
        self.sample_coords = None
        self.sample_grades = None
    
    def fit(
        self,
        coords: np.ndarray,
        grades: np.ndarray,
        group_ids: Optional[np.ndarray] = None,
        max_samples: Optional[int] = 1000,
        cv_folds: int = 5,
        compute_residuals: bool = True
    ) -> HybridModelResults:
        """
        Fit hybrid IDW+ML model to drillhole data.
        
        Args:
            coords: Sample coordinates (n_samples, 3) - [X, Y, Z]
            grades: Sample grades (n_samples,)
            group_ids: Optional group IDs for spatial cross-validation (e.g., drillhole IDs).
                      If None, uses all samples as one group.
            max_samples: Maximum number of samples to process for IDW residuals.
                        If None, processes all samples.
            cv_folds: Number of CV folds for model evaluation
            compute_residuals: If True, compute IDW residuals using leave-one-out.
                             If False, use faster approximation.
        
        Returns:
            HybridModelResults with training results
            
        Example:
            >>> coords = samples[['x', 'y', 'z']].values
            >>> grades = samples['grade'].values
            >>> hole_ids = samples['hole_id'].values
            >>> results = model.fit(coords, grades, group_ids=hole_ids)
            >>> print(f"CV MAE: {np.mean(results.cv_scores['mae']):.4f} g/t")
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        coords = np.asarray(coords, dtype=np.float64)
        grades = np.asarray(grades, dtype=np.float64)
        
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Coordinates must be 2D array with 3 columns (X, Y, Z)")
        
        if len(coords) != len(grades):
            raise ValueError("Coordinates and grades must have same length")
        
        if len(coords) == 0:
            raise ValueError("Training data cannot be empty")
        
        logger.info(f"Fitting hybrid model on {len(coords):,} samples")
        
        # Step 1: Compute IDW baseline predictions and residuals
        logger.info("Computing IDW baseline...")
        if compute_residuals:
            idw_predictions, residuals = compute_idw_residuals(
                coords, grades, k=self.idw_k, power=self.idw_power,
                max_samples=max_samples, leave_one_out=True
            )
        else:
            # Fast approximation: use full IDW
            idw_predictions = idw_interpolate(
                coords, grades, coords, k=self.idw_k, power=self.idw_power
            )
            residuals = grades - idw_predictions
        
        logger.info(
            f"IDW residuals: mean={residuals.mean():.4f}, "
            f"std={residuals.std():.4f}"
        )
        
        # Step 2: Build spatial features
        logger.info("Building spatial features...")
        features, scalers = build_spatial_features(
            coords, return_scalers=True
        )
        self.feature_scalers = scalers
        
        # Step 3: Train ML model on residuals with spatial cross-validation
        logger.info(f"Training ML model ({self.ml_model_type})...")
        
        if group_ids is None:
            # Create dummy groups (all same group)
            group_ids = np.zeros(len(coords), dtype=int)
        
        group_ids = np.asarray(group_ids)
        
        # Create ML model
        if self.ml_model_type == 'gradient_boosting':
            model_class = GradientBoostingRegressor
        elif self.ml_model_type == 'random_forest':
            model_class = RandomForestRegressor
        else:
            raise ValueError(f"Unknown model type: {self.ml_model_type}")
        
        # Cross-validation by group (e.g., by drillhole)
        gkf = GroupKFold(n_splits=min(cv_folds, len(np.unique(group_ids))))
        cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(features, residuals, group_ids)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = residuals[train_idx], residuals[test_idx]
            
            # Train model
            model = model_class(**self.ml_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Compute metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['r2'].append(r2)
            
            logger.info(f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        self.cv_scores = cv_scores
        
        logger.info(
            f"Cross-validation: MAE={np.mean(cv_scores['mae']):.4f} ± "
            f"{np.std(cv_scores['mae']):.4f}, "
            f"R²={np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}"
        )
        
        # Train final model on all data
        self.ml_model = model_class(**self.ml_params)
        self.ml_model.fit(features, residuals)
        
        # Store sample data for prediction
        self.sample_coords = coords.copy()
        self.sample_grades = grades.copy()
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.ml_model, 'feature_importances_'):
            feature_importance = self.ml_model.feature_importances_
        
        # Predict on training data for results
        ml_predictions = self.ml_model.predict(features)
        final_predictions = idw_predictions + ml_predictions
        
        logger.info("✓ Model fitted successfully")
        
        return HybridModelResults(
            idw_predictions=idw_predictions,
            residuals=residuals,
            ml_predictions=ml_predictions,
            final_predictions=final_predictions,
            cv_scores=cv_scores,
            feature_importance=feature_importance
        )
    
    def predict(
        self,
        grid_coords: np.ndarray,
        sample_coords: Optional[np.ndarray] = None,
        sample_grades: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict ore grades on block model grid.
        
        Args:
            grid_coords: Grid coordinates (n_blocks, 3) - [X, Y, Z]
            sample_coords: Optional sample coordinates for IDW (n_samples, 3).
                          If None, uses stored coordinates from fit.
            sample_grades: Optional sample grades for IDW (n_samples,).
                          If None, uses stored grades from fit.
        
        Returns:
            Tuple of (idw_grades, ml_residuals, final_grades):
                - idw_grades: IDW baseline predictions
                - ml_residuals: ML residual predictions
                - final_grades: Final fusion predictions (IDW + ML residual)
                
        Example:
            >>> grid, info = create_block_model_grid(sample_coords)
            >>> idw, residuals, final = model.predict(grid)
            >>> print(f"Final grade range: {final.min():.2f} - {final.max():.2f} g/t")
        """
        if self.ml_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.feature_scalers is None:
            raise ValueError("Feature scalers not available. Model may not be fitted.")
        
        grid_coords = np.asarray(grid_coords, dtype=np.float64)
        
        if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
            raise ValueError("Grid coordinates must be 2D array with 3 columns")
        
        # Use stored sample data if not provided
        if sample_coords is None:
            if self.sample_coords is None:
                raise ValueError("sample_coords required (not stored during fit)")
            sample_coords = self.sample_coords
        
        if sample_grades is None:
            if self.sample_grades is None:
                raise ValueError("sample_grades required (not stored during fit)")
            sample_grades = self.sample_grades
        
        sample_coords = np.asarray(sample_coords, dtype=np.float64)
        sample_grades = np.asarray(sample_grades, dtype=np.float64)
        
        logger.info(f"Predicting grades on {len(grid_coords):,} blocks...")
        
        # Step 1: IDW baseline prediction
        logger.info("Computing IDW baseline...")
        idw_grades = idw_interpolate(
            sample_coords, sample_grades, grid_coords,
            k=self.idw_k, power=self.idw_power
        )
        
        # Step 2: Build grid features
        logger.info("Building grid features...")
        grid_features = build_block_model_features(
            grid_coords, sample_coords, self.feature_scalers
        )
        
        # Step 3: Predict ML residuals
        logger.info("Predicting ML residuals...")
        ml_residuals = self.ml_model.predict(grid_features)
        
        # Step 4: Fusion (IDW + ML residual)
        final_grades = idw_grades + ml_residuals
        
        logger.info(
            f"Predictions: IDW range=[{idw_grades.min():.4f}, {idw_grades.max():.4f}], "
            f"Final range=[{final_grades.min():.4f}, {final_grades.max():.4f}]"
        )
        
        return idw_grades, ml_residuals, final_grades


def train_hybrid_model(
    coords: np.ndarray,
    grades: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    idw_k: int = 16,
    idw_power: float = 2.0,
    ml_model_type: str = 'gradient_boosting',
    cv_folds: int = 5
) -> HybridOreModel:
    """
    Convenience function to train hybrid IDW+ML model.
    
    Args:
        coords: Sample coordinates (n_samples, 3)
        grades: Sample grades (n_samples,)
        group_ids: Optional group IDs for CV (e.g., drillhole IDs)
        idw_k: Number of neighbors for IDW
        idw_power: IDW exponent
        ml_model_type: ML model type
        cv_folds: Number of CV folds
        
    Returns:
        Fitted HybridOreModel
        
    Example:
        >>> coords = samples[['x', 'y', 'z']].values
        >>> grades = samples['grade'].values
        >>> model = train_hybrid_model(coords, grades, group_ids=hole_ids)
    """
    model = HybridOreModel(
        idw_k=idw_k,
        idw_power=idw_power,
        ml_model_type=ml_model_type
    )
    
    model.fit(coords, grades, group_ids=group_ids, cv_folds=cv_folds)
    
    return model


def predict_block_grades(
    model: HybridOreModel,
    grid_coords: np.ndarray,
    sample_coords: np.ndarray,
    sample_grades: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict block grades using hybrid model.
    
    Complete prediction including IDW baseline.
    
    Args:
        model: Fitted HybridOreModel
        grid_coords: Grid coordinates (n_blocks, 3)
        sample_coords: Sample coordinates (n_samples, 3)
        sample_grades: Sample grades (n_samples,)
        
    Returns:
        Tuple of (idw_grades, ml_residuals, final_grades)
    """
    if model.ml_model is None:
        raise ValueError("Model must be fitted")
    
    # IDW baseline
    idw_grades = idw_interpolate(
        sample_coords, sample_grades, grid_coords,
        k=model.idw_k, power=model.idw_power
    )
    
    # ML residuals
    grid_features = build_block_model_features(
        grid_coords, sample_coords, model.feature_scalers
    )
    ml_residuals = model.ml_model.predict(grid_features)
    
    # Fusion
    final_grades = idw_grades + ml_residuals
    
    return idw_grades, ml_residuals, final_grades

