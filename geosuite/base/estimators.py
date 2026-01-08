"""
Base classes for estimators following scikit-learn conventions.
"""
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseEstimator(ABC):
    """
    Base class for estimators in GeoSuite.
    
    Estimators learn from data and make predictions
    (e.g., ML models, regression models).
    
    Example:
        >>> from geosuite.base import BaseEstimator
        >>> class PermeabilityEstimator(BaseEstimator):
        ...     def fit(self, X, y):
        ...         # Train model
        ...         self.model_ = RandomForestRegressor()
        ...         self.model_.fit(X, y)
        ...         return self
        ...     
        ...     def predict(self, X):
        ...         return self.model_.predict(X)
        >>> 
        >>> estimator = PermeabilityEstimator()
        >>> estimator.fit(X_train, y_train)
        >>> predictions = estimator.predict(X_test)
    """
    
    def __init__(self):
        """Initialize the estimator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'BaseEstimator':
        """
        Fit the estimator to the data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training data (features)
        y : np.ndarray or pd.Series
            Target values
            
        Returns
        -------
        BaseEstimator
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series]:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data
            
        Returns
        -------
        np.ndarray or pd.Series
            Predictions
        """
        pass
    
    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Return the score (e.g., R² for regression, accuracy for classification).
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Test data
        y : np.ndarray or pd.Series
            True target values
            
        Returns
        -------
        float
            Score metric
        """
        from sklearn.metrics import r2_score, accuracy_score
        
        y_pred = self.predict(X)
        
        # Determine if regression or classification
        score_funcs = {
            'classifier': accuracy_score,
            'regressor': r2_score,
        }
        
        estimator_type = getattr(self, '_estimator_type', 'regressor')
        score_func = score_funcs.get(estimator_type, r2_score)
        
        return score_func(y, y_pred)
    
    def _validate_fit_input(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert fit inputs.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Features
        y : np.ndarray or pd.Series
            Targets
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Validated (X, y) as numpy arrays
            
        Raises
        ------
        ValueError
            If inputs are invalid or mismatched
        """
        type_converters = {
            pd.DataFrame: lambda x: x.values,
            pd.Series: lambda x: x.values,
            np.ndarray: lambda x: x,
        }
        
        converter_x = type_converters.get(type(X))
        if converter_x is None:
            raise ValueError("X must be np.ndarray or pd.DataFrame")
        X = converter_x(X)
        
        converter_y = type_converters.get(type(y))
        if converter_y is None:
            raise ValueError("y must be np.ndarray or pd.Series")
        y = converter_y(y)
        
        if len(X) == 0:
            raise ValueError("X must not be empty")
        
        if len(y) == 0:
            raise ValueError("y must not be empty")
        
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. "
                f"Got X: {len(X)}, y: {len(y)}"
            )
        
        return X, y

