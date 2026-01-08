"""
Base classes for data transformers following scikit-learn conventions.
"""
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """
    Base class for data transformers in GeoSuite.
    
    Follows scikit-learn conventions with fit/transform methods.
    Transformers modify input data (e.g., preprocessing, normalization).
    
    Example:
        >>> from geosuite.base import BaseTransformer
        >>> class LogNormalizer(BaseTransformer):
        ...     def fit(self, X, y=None):
        ...         self.mean_ = X.mean()
        ...         self.std_ = X.std()
        ...         return self
        ...     
        ...     def transform(self, X):
        ...         return (X - self.mean_) / self.std_
        >>> 
        >>> transformer = LogNormalizer()
        >>> transformer.fit(df['GR'])
        >>> normalized = transformer.transform(df['GR'])
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'BaseTransformer':
        """
        Fit the transformer to the data.
        
        Parameters
        ----------
        X : np.ndarray, pd.Series, or pd.DataFrame
            Input data
        y : np.ndarray or pd.Series, optional
            Target values (unused for most transformers)
            
        Returns
        -------
        BaseTransformer
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        X: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Transform the data.
        
        Parameters
        ----------
        X : np.ndarray, pd.Series, or pd.DataFrame
            Input data to transform
            
        Returns
        -------
        np.ndarray, pd.Series, or pd.DataFrame
            Transformed data
        """
        pass
    
    def fit_transform(
        self,
        X: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Fit the transformer and transform the data in one step.
        
        Parameters
        ----------
        X : np.ndarray, pd.Series, or pd.DataFrame
            Input data
        y : np.ndarray or pd.Series, optional
            Target values
            
        Returns
        -------
        np.ndarray, pd.Series, or pd.DataFrame
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(
        self,
        X: Union[np.ndarray, pd.Series, pd.DataFrame],
        name: str = "X"
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Validate input data.
        
        Parameters
        ----------
        X : np.ndarray, pd.Series, or pd.DataFrame
            Input data to validate
        name : str, default "X"
            Name of the input for error messages
            
        Returns
        -------
        np.ndarray, pd.Series, or pd.DataFrame
            Validated input
            
        Raises
        ------
        ValueError
            If input is empty or invalid
        """
        validation_checks = {
            pd.DataFrame: lambda x: x.empty,
            pd.Series: lambda x: len(x) == 0,
            np.ndarray: lambda x: len(x) == 0,
        }
        
        for valid_type, check_func in validation_checks.items():
            if isinstance(X, valid_type):
                if check_func(X):
                    raise ValueError(f"{name} must not be empty")
                return X
        
        raise ValueError(
            f"{name} must be np.ndarray, pd.Series, or pd.DataFrame, "
            f"got {type(X)}"
        )
        
        return X
    
    def _get_feature_names_out(self) -> list:
        """
        Get output feature names for transformers.
        
        Returns
        -------
        list
            List of feature names
        """
        if hasattr(self, 'feature_names_in_'):
            return list(self.feature_names_in_)
        return []

