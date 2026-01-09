"""
Time series decomposition for production data.

Provides trend and seasonality detection and removal for production forecasting.
"""
import logging
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


def decompose_production(
    time: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
    method: str = 'moving_average',
    seasonal_period: Optional[int] = None,
    trend_window: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Decompose production time series into trend, seasonal, and residual components.
    
    Parameters
    ----------
    time : np.ndarray or pd.Series
        Time array
    production : np.ndarray or pd.Series
        Production values
    method : str, default 'moving_average'
        Decomposition method: 'moving_average' or 'stl'
    seasonal_period : int, optional
        Seasonal period (auto-detected if not specified)
    trend_window : int, optional
        Window size for trend extraction (auto-determined if not specified)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'trend': Trend component
        - 'seasonal': Seasonal component
        - 'residual': Residual component
        - 'original': Original data
    """
    time_arr = np.asarray(time, dtype=float)
    prod_arr = np.asarray(production, dtype=float)
    
    # Remove invalid values
    valid_mask = np.isfinite(prod_arr) & np.isfinite(time_arr)
    time_arr = time_arr[valid_mask]
    prod_arr = prod_arr[valid_mask]
    
    if len(prod_arr) < 10:
        raise ValueError("Need at least 10 data points for decomposition")
    
    if method == 'moving_average':
        return _decompose_moving_average(prod_arr, seasonal_period, trend_window)
    elif method == 'stl':
        return _decompose_stl(prod_arr, seasonal_period, trend_window)
    else:
        raise ValueError(f"Unknown method: {method}")


def _decompose_moving_average(
    production: np.ndarray,
    seasonal_period: Optional[int],
    trend_window: Optional[int]
) -> Dict[str, np.ndarray]:
    """Decompose using moving average method."""
    n = len(production)
    
    # Auto-determine trend window
    if trend_window is None:
        trend_window = max(3, n // 10)
    
    # Extract trend using moving average
    trend = uniform_filter1d(production, size=trend_window, mode='nearest')
    
    # Detrend
    detrended = production - trend
    
    # Extract seasonal component
    if seasonal_period is None:
        seasonal_period = _detect_seasonal_period(detrended)
    
    seasonal = np.zeros_like(production)
    if seasonal_period and seasonal_period > 1:
        # Average over seasonal periods
        n_periods = n // seasonal_period
        if n_periods > 0:
            seasonal_pattern = np.zeros(seasonal_period)
            for i in range(seasonal_period):
                indices = np.arange(i, n, seasonal_period)
                if len(indices) > 0:
                    seasonal_pattern[i] = np.mean(detrended[indices])
            
            # Center seasonal pattern
            seasonal_pattern = seasonal_pattern - np.mean(seasonal_pattern)
            
            # Replicate pattern
            for i in range(n):
                seasonal[i] = seasonal_pattern[i % seasonal_period]
    
    # Residual
    residual = detrended - seasonal
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
        'original': production
    }


def _decompose_stl(
    production: np.ndarray,
    seasonal_period: Optional[int],
    trend_window: Optional[int]
) -> Dict[str, np.ndarray]:
    """Decompose using STL-like method (simplified)."""
    # For now, use moving average as STL requires statsmodels
    # Could be enhanced with statsmodels.tsa.seasonal.STL if available
    return _decompose_moving_average(production, seasonal_period, trend_window)


def _detect_seasonal_period(data: np.ndarray, max_period: int = 50) -> Optional[int]:
    """
    Detect seasonal period using autocorrelation.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    max_period : int, default 50
        Maximum period to check
        
    Returns
    -------
    int or None
        Detected seasonal period
    """
    n = len(data)
    if n < max_period * 2:
        return None
    
    # Compute autocorrelation
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[n-1:] / autocorr[n-1]
    
    # Find peaks in autocorrelation (potential seasonal periods)
    peaks, _ = signal.find_peaks(autocorr[1:max_period], height=0.3)
    
    if len(peaks) > 0:
        # Return first significant peak
        return int(peaks[0] + 1)
    
    return None


def detect_trend(
    production: Union[np.ndarray, pd.Series],
    method: str = 'linear'
) -> Dict[str, Any]:
    """
    Detect trend in production data.
    
    Parameters
    ----------
    production : np.ndarray or pd.Series
        Production values
    method : str, default 'linear'
        Trend detection method: 'linear', 'polynomial', or 'moving_average'
        
    Returns
    -------
    dict
        Trend information including:
        - 'trend': Trend values
        - 'slope': Trend slope (for linear)
        - 'strength': Trend strength (0-1)
    """
    prod_arr = np.asarray(production, dtype=float)
    valid_mask = np.isfinite(prod_arr)
    prod_arr = prod_arr[valid_mask]
    
    if len(prod_arr) < 3:
        raise ValueError("Need at least 3 data points")
    
    time_arr = np.arange(len(prod_arr))
    
    if method == 'linear':
        slope, intercept, r_value, _, _ = np.polyfit(time_arr, prod_arr, 1, full=False)
        trend = slope * time_arr + intercept
        strength = abs(r_value)
        
        return {
            'trend': trend,
            'slope': float(slope),
            'intercept': float(intercept),
            'strength': float(strength)
        }
    
    elif method == 'polynomial':
        coeffs = np.polyfit(time_arr, prod_arr, deg=2)
        trend = np.polyval(coeffs, time_arr)
        # Calculate R-squared as strength
        ss_res = np.sum((prod_arr - trend) ** 2)
        ss_tot = np.sum((prod_arr - np.mean(prod_arr)) ** 2)
        strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'trend': trend,
            'coefficients': coeffs.tolist(),
            'strength': float(strength)
        }
    
    elif method == 'moving_average':
        window = max(3, len(prod_arr) // 10)
        trend = uniform_filter1d(prod_arr, size=window, mode='nearest')
        # Calculate trend strength as variance reduction
        var_original = np.var(prod_arr)
        var_residual = np.var(prod_arr - trend)
        strength = 1 - (var_residual / var_original) if var_original > 0 else 0
        
        return {
            'trend': trend,
            'strength': float(strength)
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_seasonality(
    production: Union[np.ndarray, pd.Series],
    max_period: int = 50
) -> Dict[str, Any]:
    """
    Detect seasonality in production data.
    
    Parameters
    ----------
    production : np.ndarray or pd.Series
        Production values
    max_period : int, default 50
        Maximum period to check
        
    Returns
    -------
    dict
        Seasonality information including:
        - 'period': Detected seasonal period
        - 'strength': Seasonality strength (0-1)
        - 'pattern': Seasonal pattern
    """
    prod_arr = np.asarray(production, dtype=float)
    valid_mask = np.isfinite(prod_arr)
    prod_arr = prod_arr[valid_mask]
    
    if len(prod_arr) < max_period * 2:
        return {
            'period': None,
            'strength': 0.0,
            'pattern': None
        }
    
    # Remove trend first
    trend_info = detect_trend(prod_arr, method='linear')
    detrended = prod_arr - trend_info['trend']
    
    # Detect seasonal period
    period = _detect_seasonal_period(detrended, max_period)
    
    if period is None:
        return {
            'period': None,
            'strength': 0.0,
            'pattern': None
        }
    
    # Extract seasonal pattern
    n = len(detrended)
    n_periods = n // period
    seasonal_pattern = np.zeros(period)
    
    for i in range(period):
        indices = np.arange(i, n, period)
        if len(indices) > 0:
            seasonal_pattern[i] = np.mean(detrended[indices])
    
    # Center pattern
    seasonal_pattern = seasonal_pattern - np.mean(seasonal_pattern)
    
    # Calculate strength as variance explained
    seasonal_component = np.tile(seasonal_pattern, n_periods + 1)[:n]
    var_seasonal = np.var(seasonal_component)
    var_total = np.var(detrended)
    strength = var_seasonal / var_total if var_total > 0 else 0.0
    
    return {
        'period': int(period),
        'strength': float(strength),
        'pattern': seasonal_pattern.tolist()
    }


def remove_trend_seasonality(
    time: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
    remove_trend: bool = True,
    remove_seasonal: bool = True
) -> np.ndarray:
    """
    Remove trend and/or seasonality from production data.
    
    Parameters
    ----------
    time : np.ndarray or pd.Series
        Time array
    production : np.ndarray or pd.Series
        Production values
    remove_trend : bool, default True
        Whether to remove trend
    remove_seasonal : bool, default True
        Whether to remove seasonality
        
    Returns
    -------
    np.ndarray
        Production data with trend/seasonality removed
    """
    decomposed = decompose_production(time, production)
    
    result = decomposed['original'].copy()
    
    if remove_trend:
        result = result - decomposed['trend']
    
    if remove_seasonal:
        result = result - decomposed['seasonal']
    
    return result


