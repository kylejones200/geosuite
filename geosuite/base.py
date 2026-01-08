"""
Base classes for GeoSuite API consistency.

Provides scikit-learn-style base classes for transformers and calculators
to ensure consistent API patterns across the library.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """
    Base class for data transformers following scikit-learn conventions.
    
    Transformers modify input data (e.g., preprocessing, feature engineering)
    and should implement fit/transform or fit_transform methods.
    
    Example:
        >>> from geosuite.base import BaseTransformer
        >>> class LogNormalizer(BaseTransformer):
        ...     def fit(self, X, y=None):
        ...         self.mean_ = X.mean()
        ...         return self
        ...     def transform(self, X):
        ...         return np.log(X - self.mean_ + 1)
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'BaseTransformer':
        """
        Fit the transformer to the data.
        
        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or pd.Series
            Input data
        y : np.ndarray or pd.Series, optional
            Target values (unused for most transformers)
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Transform the input data.
        
        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or pd.Series
            Input data to transform
            
        Returns
        -------
        np.ndarray, pd.DataFrame, or pd.Series
            Transformed data
        """
        pass
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame, pd.Series],
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Fit the transformer and transform the data in one step.
        
        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or pd.Series
            Input data
        y : np.ndarray or pd.Series, optional
            Target values
            
        Returns
        -------
        np.ndarray, pd.DataFrame, or pd.Series
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def _check_is_fitted(self):
        """Check if transformer has been fitted."""
        if not self._is_fitted:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' before 'transform'."
            )


class BaseCalculator(ABC):
    """
    Base class for calculation functions following consistent patterns.
    
    Calculators perform computations on input data and return results.
    They may or may not require fitting (some are stateless).
    
    Example:
        >>> from geosuite.base import BaseCalculator
        >>> class PorosityCalculator(BaseCalculator):
        ...     def calculate(self, density, rho_matrix=2.65, rho_fluid=1.0):
        ...         return (rho_matrix - density) / (rho_matrix - rho_fluid)
    """
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    @abstractmethod
    def calculate(self, *args, **kwargs) -> Union[np.ndarray, pd.Series, float]:
        """
        Perform the calculation.
        
        Returns
        -------
        np.ndarray, pd.Series, or float
            Calculation results
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Union[np.ndarray, pd.Series, float]:
        """
        Allow calculator to be called directly.
        
        Returns
        -------
        np.ndarray, pd.Series, or float
            Calculation results
        """
        return self.calculate(*args, **kwargs)


class BaseEstimator(ABC):
    """
    Base class for estimators (models) following scikit-learn conventions.
    
    Estimators learn from data and make predictions. They implement
    fit/predict or fit_predict methods.
    
    Example:
        >>> from geosuite.base import BaseEstimator
        >>> class SimpleRegressor(BaseEstimator):
        ...     def fit(self, X, y):
        ...         self.coef_ = np.linalg.lstsq(X, y)[0]
        ...         return self
        ...     def predict(self, X):
        ...         return X @ self.coef_
    """
    
    def __init__(self):
        """Initialize the estimator."""
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'BaseEstimator':
        """
        Fit the estimator to the training data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray or pd.Series
            Predictions
        """
        pass
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame],
                   y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Fit the estimator and make predictions in one step.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets
            
        Returns
        -------
        np.ndarray or pd.Series
            Predictions
        """
        return self.fit(X, y).predict(X)
    
    def _check_is_fitted(self):
        """Check if estimator has been fitted."""
        if not self._is_fitted:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' before 'predict'."
            )
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def set_params(self, **params) -> 'BaseEstimator':
        """
        Set parameters for this estimator.
        
        Parameters
        ----------
        **params
            Parameter names and values
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key} for {self.__class__.__name__}")
            setattr(self, key, value)
        return self

