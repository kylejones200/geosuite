"""
Regression models for petrophysical property prediction.

Provides pipelines for predicting continuous properties like permeability
and porosity from well log data.
"""
import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..base.estimators import BaseEstimator

logger = logging.getLogger(__name__)


class PermeabilityPredictor(BaseEstimator):
    """
    Predict permeability from well log data.
    
    Uses machine learning regression models to predict permeability
    from petrophysical properties and well logs.
    
    Example:
        >>> from geosuite.ml import PermeabilityPredictor
        >>> 
        >>> predictor = PermeabilityPredictor(model_type='random_forest')
        >>> predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_test)
        >>> score = predictor.score(X_test, y_test)
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params
    ):
        """
        Initialize the permeability predictor.
        
        Parameters
        ----------
        model_type : str, default 'random_forest'
            Type of model: 'random_forest', 'gradient_boosting', 'ridge', 'lasso'
        test_size : float, default 0.2
            Fraction of data to use for testing
        random_state : int, default 42
            Random seed for reproducibility
        **model_params
            Additional parameters passed to the model
        """
        super().__init__()
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
        self.model = None
        self.scaler = None
        self._estimator_type = 'regressor'
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'PermeabilityPredictor':
        """
        Fit the permeability prediction model.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features (well log data)
        y : np.ndarray or pd.Series
            Training targets (permeability values)
            
        Returns
        -------
        PermeabilityPredictor
            Self for method chaining
        """
        X, y = self._validate_fit_input(X, y)
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Split data if test_size > 0
        if self.test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Create model pipeline
        model_configs = {
            'random_forest': {
                'default_params': {'n_estimators': 300, 'max_depth': None, 'random_state': self.random_state},
                'regressor': RandomForestRegressor,
            },
            'gradient_boosting': {
                'default_params': {'n_estimators': 200, 'max_depth': 5, 'random_state': self.random_state},
                'regressor': GradientBoostingRegressor,
            },
            'ridge': {
                'default_params': {'alpha': 1.0, 'random_state': self.random_state},
                'regressor': Ridge,
            },
            'lasso': {
                'default_params': {'alpha': 1.0, 'random_state': self.random_state},
                'regressor': Lasso,
            },
        }
        
        if self.model_type not in model_configs:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Supported: {', '.join(model_configs.keys())}"
            )
        
        config = model_configs[self.model_type]
        default_params = {**config['default_params'], **self.model_params}
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', config['regressor'](**default_params))
        ])
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set if available
        if X_test is not None:
            y_pred = self.model.predict(X_test)
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
            }
            self.logger.info(
                f"Fitted {self.model_type} model: "
                f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}"
            )
        else:
            self.logger.info(f"Fitted {self.model_type} model on full dataset")
        
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict permeability values.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features (well log data)
            
        Returns
        -------
        np.ndarray
            Predicted permeability values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        predictions = self.model.predict(X)
        
        # Ensure non-negative (permeability can't be negative)
        predictions = np.maximum(predictions, 0)
        
        return predictions


class PorosityPredictor(BaseEstimator):
    """
    Predict porosity from well log data.
    
    Similar to PermeabilityPredictor but for porosity prediction.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params
    ):
        """
        Initialize the porosity predictor.
        
        Parameters
        ----------
        model_type : str, default 'random_forest'
            Type of model to use
        test_size : float, default 0.2
            Fraction of data for testing
        random_state : int, default 42
            Random seed
        **model_params
            Additional model parameters
        """
        super().__init__()
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
        self.model = None
        self._estimator_type = 'regressor'
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'PorosityPredictor':
        """
        Fit the porosity prediction model.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets (porosity values)
            
        Returns
        -------
        PorosityPredictor
            Self for method chaining
        """
        # Use PermeabilityPredictor implementation (same structure)
        predictor = PermeabilityPredictor(
            model_type=self.model_type,
            test_size=self.test_size,
            random_state=self.random_state,
            **self.model_params
        )
        predictor.fit(X, y)
        self.model = predictor.model
        
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict porosity values.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Predicted porosity values (clipped to [0, 1])
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        predictions = self.model.predict(X)
        
        # Clip to valid porosity range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
