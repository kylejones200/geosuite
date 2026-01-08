"""
Physics-informed decline curve models.

Implements standard decline curve models (exponential, hyperbolic, harmonic)
with physics-informed constraints and parameter estimation.
"""
import logging
from typing import Optional, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class DeclineModel(ABC):
    """
    Base class for decline curve models.
    
    All decline models follow the same interface for fitting and forecasting.
    """
    
    def __init__(self):
        """Initialize decline model."""
        self.params = {}
        self.fitted = False
    
    @abstractmethod
    def _rate_function(self, t: np.ndarray, *params) -> np.ndarray:
        """
        Rate function for the decline model.
        
        Parameters
        ----------
        t : np.ndarray
            Time array
        *params
            Model parameters
            
        Returns
        -------
        np.ndarray
            Production rate
        """
        pass
    
    @abstractmethod
    def _cumulative_function(self, t: np.ndarray, *params) -> np.ndarray:
        """
        Cumulative production function.
        
        Parameters
        ----------
        t : np.ndarray
            Time array
        *params
            Model parameters
            
        Returns
        -------
        np.ndarray
            Cumulative production
        """
        pass
    
    def fit(
        self,
        time: Union[np.ndarray, pd.Series],
        rate: Union[np.ndarray, pd.Series],
        initial_params: Optional[Dict[str, float]] = None
    ) -> 'DeclineModel':
        """
        Fit decline model to production data.
        
        Parameters
        ----------
        time : np.ndarray or pd.Series
            Time array (days, months, etc.)
        rate : np.ndarray or pd.Series
            Production rate (volume/time)
        initial_params : dict, optional
            Initial parameter guesses
            
        Returns
        -------
        self
        """
        time_arr = np.asarray(time, dtype=float)
        rate_arr = np.asarray(rate, dtype=float)
        
        # Remove invalid values
        valid_mask = (rate_arr > 0) & np.isfinite(rate_arr) & np.isfinite(time_arr)
        time_arr = time_arr[valid_mask]
        rate_arr = rate_arr[valid_mask]
        
        if len(time_arr) < 3:
            raise ValueError("Need at least 3 valid data points to fit decline model")
        
        # Normalize time to start at 0
        t0 = time_arr[0]
        time_normalized = time_arr - t0
        
        # Fit model
        try:
            if initial_params is None:
                initial_params = self._estimate_initial_params(time_normalized, rate_arr)
            
            popt, _ = curve_fit(
                self._rate_function,
                time_normalized,
                rate_arr,
                p0=list(initial_params.values()),
                bounds=self._get_bounds(),
                maxfev=10000
            )
            
            # Store parameters
            param_names = list(initial_params.keys())
            self.params = dict(zip(param_names, popt))
            self.fitted = True
            
            logger.info(f"Fitted {self.__class__.__name__} with params: {self.params}")
            
        except Exception as e:
            logger.error(f"Error fitting decline model: {e}")
            raise
        
        return self
    
    @abstractmethod
    def _estimate_initial_params(
        self,
        time: np.ndarray,
        rate: np.ndarray
    ) -> Dict[str, float]:
        """Estimate initial parameters from data."""
        pass
    
    @abstractmethod
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for optimization."""
        pass
    
    def predict(
        self,
        time: Union[np.ndarray, pd.Series],
        return_cumulative: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict production rate (and optionally cumulative) for given times.
        
        Parameters
        ----------
        time : np.ndarray or pd.Series
            Time array for prediction
        return_cumulative : bool, default False
            If True, also return cumulative production
            
        Returns
        -------
        np.ndarray or tuple
            Production rate (and cumulative if requested)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        time_arr = np.asarray(time, dtype=float)
        time_normalized = time_arr - time_arr[0]
        
        rate = self._rate_function(time_normalized, *list(self.params.values()))
        
        if return_cumulative:
            cumulative = self._cumulative_function(time_normalized, *list(self.params.values()))
            return rate, cumulative
        
        return rate
    
    def forecast(
        self,
        n_periods: int,
        period_length: float = 1.0,
        start_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Forecast production for future periods.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to forecast
        period_length : float, default 1.0
            Length of each period (same units as training data)
        start_time : float, optional
            Start time for forecast (uses last training time if not specified)
            
        Returns
        -------
        pd.DataFrame
            Forecast with columns: time, rate, cumulative
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if start_time is None:
            # Use last fitted time
            start_time = 0.0  # Will be normalized
        
        forecast_times = np.arange(n_periods) * period_length + start_time
        rate, cumulative = self.predict(forecast_times, return_cumulative=True)
        
        return pd.DataFrame({
            'time': forecast_times,
            'rate': rate,
            'cumulative': cumulative
        })


class ExponentialDecline(DeclineModel):
    """
    Exponential decline model: q(t) = q_i * exp(-D_i * t)
    
    Where:
    - q_i: Initial production rate
    - D_i: Decline rate (1/time)
    """
    
    def _rate_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Exponential decline rate function."""
        return qi * np.exp(-di * t)
    
    def _cumulative_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Exponential decline cumulative function."""
        return (qi / di) * (1 - np.exp(-di * t))
    
    def _estimate_initial_params(self, time: np.ndarray, rate: np.ndarray) -> Dict[str, float]:
        """Estimate initial parameters for exponential decline."""
        # Use linear regression on log(rate) vs time
        valid_mask = rate > 0
        if np.sum(valid_mask) < 2:
            return {'qi': rate[0], 'di': 0.01}
        
        log_rate = np.log(rate[valid_mask])
        time_valid = time[valid_mask]
        
        slope, intercept, _, _, _ = linregress(time_valid, log_rate)
        
        qi = np.exp(intercept)
        di = -slope if slope < 0 else 0.01
        
        return {'qi': float(qi), 'di': float(di)}
    
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for exponential decline."""
        return (
            np.array([0.0, 0.0]),  # qi > 0, di > 0
            np.array([np.inf, np.inf])
        )


class HyperbolicDecline(DeclineModel):
    """
    Hyperbolic decline model: q(t) = q_i / (1 + b * D_i * t)^(1/b)
    
    Where:
    - q_i: Initial production rate
    - D_i: Initial decline rate (1/time)
    - b: Hyperbolic exponent (0 < b < 1)
    """
    
    def _rate_function(self, t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
        """Hyperbolic decline rate function."""
        return qi / np.power(1 + b * di * t, 1.0 / b)
    
    def _cumulative_function(self, t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
        """Hyperbolic decline cumulative function."""
        if b == 1.0:
            # Harmonic case
            return (qi / di) * np.log(1 + di * t)
        else:
            return (qi / ((1 - b) * di)) * (1 - np.power(1 + b * di * t, (b - 1) / b))
    
    def _estimate_initial_params(self, time: np.ndarray, rate: np.ndarray) -> Dict[str, float]:
        """Estimate initial parameters for hyperbolic decline."""
        qi = rate[0]
        di = 0.01
        b = 0.5  # Typical value
        
        return {'qi': float(qi), 'di': float(di), 'b': float(b)}
    
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for hyperbolic decline."""
        return (
            np.array([0.0, 0.0, 0.01]),  # qi > 0, di > 0, 0.01 < b < 1
            np.array([np.inf, np.inf, 0.99])
        )


class HarmonicDecline(DeclineModel):
    """
    Harmonic decline model: q(t) = q_i / (1 + D_i * t)
    
    Where:
    - q_i: Initial production rate
    - D_i: Decline rate (1/time)
    
    This is a special case of hyperbolic decline with b = 1.
    """
    
    def _rate_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Harmonic decline rate function."""
        return qi / (1 + di * t)
    
    def _cumulative_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Harmonic decline cumulative function."""
        return (qi / di) * np.log(1 + di * t)
    
    def _estimate_initial_params(self, time: np.ndarray, rate: np.ndarray) -> Dict[str, float]:
        """Estimate initial parameters for harmonic decline."""
        qi = rate[0]
        # Estimate di from rate ratio
        if len(rate) > 1 and rate[-1] > 0:
            di = (qi / rate[-1] - 1) / time[-1] if time[-1] > 0 else 0.01
        else:
            di = 0.01
        
        return {'qi': float(qi), 'di': float(di)}
    
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for harmonic decline."""
        return (
            np.array([0.0, 0.0]),  # qi > 0, di > 0
            np.array([np.inf, np.inf])
        )


def fit_decline_model(
    time: Union[np.ndarray, pd.Series],
    rate: Union[np.ndarray, pd.Series],
    model_type: str = 'hyperbolic'
) -> DeclineModel:
    """
    Fit a decline curve model to production data.
    
    Parameters
    ----------
    time : np.ndarray or pd.Series
        Time array
    rate : np.ndarray or pd.Series
        Production rate
    model_type : str, default 'hyperbolic'
        Type of decline model: 'exponential', 'hyperbolic', or 'harmonic'
        
    Returns
    -------
    DeclineModel
        Fitted decline model
    """
    model_map = {
        'exponential': ExponentialDecline,
        'hyperbolic': HyperbolicDecline,
        'harmonic': HarmonicDecline
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_map.keys())}")
    
    model = model_map[model_type]()
    model.fit(time, rate)
    
    return model


def forecast_production(
    model: DeclineModel,
    n_periods: int,
    period_length: float = 1.0
) -> pd.DataFrame:
    """
    Forecast production using a fitted decline model.
    
    Parameters
    ----------
    model : DeclineModel
        Fitted decline model
    n_periods : int
        Number of periods to forecast
    period_length : float, default 1.0
        Length of each period
        
    Returns
    -------
    pd.DataFrame
        Forecast results
    """
    return model.forecast(n_periods, period_length)

