"""
Base classes for calculators following scikit-learn conventions.
"""
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseCalculator(ABC):
    """
    Base class for calculators in GeoSuite.
    
    Calculators compute derived quantities from input data
    (e.g., petrophysical properties, geomechanical stresses).
    
    Example:
        >>> from geosuite.base import BaseCalculator
        >>> class PorosityCalculator(BaseCalculator):
        ...     def __init__(self, rho_matrix=2.65, rho_fluid=1.0):
        ...         self.rho_matrix = rho_matrix
        ...         self.rho_fluid = rho_fluid
        ...     
        ...     def calculate(self, rhob):
        ...         return (self.rho_matrix - rhob) / (self.rho_matrix - self.rho_fluid)
        >>> 
        >>> calc = PorosityCalculator()
        >>> porosity = calc.calculate(df['RHOB'])
    """
    
    def __init__(self):
        """Initialize the calculator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def calculate(
        self,
        *args,
        **kwargs
    ) -> Union[np.ndarray, pd.Series, Dict[str, Any]]:
        """
        Perform the calculation.
        
        Parameters
        ----------
        *args
            Positional arguments (typically input data)
        **kwargs
            Keyword arguments (typically parameters)
            
        Returns
        -------
        np.ndarray, pd.Series, or Dict
            Calculated results
        """
        pass
    
    def _validate_input(
        self,
        X: Union[np.ndarray, pd.Series],
        name: str = "input"
    ) -> np.ndarray:
        """
        Validate and convert input to numpy array.
        
        Parameters
        ----------
        X : np.ndarray or pd.Series
            Input data
        name : str, default "input"
            Name for error messages
            
        Returns
        -------
        np.ndarray
            Validated numpy array
            
        Raises
        ------
        ValueError
            If input is empty or invalid
        """
        type_converters = {
            pd.Series: lambda x: x.values,
            np.ndarray: lambda x: x,
        }
        
        converter = type_converters.get(type(X))
        if converter is None:
            raise ValueError(
                f"{name} must be np.ndarray or pd.Series, got {type(X)}"
            )
        
        X = converter(X)
        
        if len(X) == 0:
            raise ValueError(f"{name} must not be empty")
        
        return X
    
    def _validate_arrays_match(
        self,
        *arrays: np.ndarray,
        names: Optional[list] = None
    ) -> None:
        """
        Validate that multiple arrays have the same length.
        
        Parameters
        ----------
        *arrays : np.ndarray
            Arrays to validate
        names : list, optional
            Names for error messages
            
        Raises
        ------
        ValueError
            If arrays have mismatched lengths
        """
        if not arrays:
            return
        
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            if names:
                name_str = ", ".join(names)
            else:
                name_str = "arrays"
            raise ValueError(
                f"All {name_str} must have the same length. "
                f"Got lengths: {lengths}"
            )

