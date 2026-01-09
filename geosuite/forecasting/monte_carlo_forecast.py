"""
Monte Carlo ensemble forecasting.

Provides Monte Carlo methods for production forecasting with uncertainty
quantification through ensemble methods.
"""
import logging
from typing import Optional, Union, Dict, List, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloForecaster:
    """
    Monte Carlo forecaster for production with uncertainty.
    
    Generates ensemble forecasts by sampling from parameter distributions
    and aggregating results to provide uncertainty bands.
    
    Example:
        >>> forecaster = MonteCarloForecaster()
        >>> ensemble = forecaster.forecast_ensemble(
        ...     time=time,
        ...     rate=rate,
        ...     n_periods=12,
        ...     n_samples=1000
        ... )
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize Monte Carlo forecaster.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def forecast_ensemble(
        self,
        time: Union[np.ndarray, pd.Series],
        rate: Union[np.ndarray, pd.Series],
        n_periods: int = 12,
        period_length: float = 1.0,
        model_type: str = 'hyperbolic',
        n_samples: int = 1000,
        parameter_uncertainty: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Generate ensemble forecast using Monte Carlo sampling.
        
        Parameters
        ----------
        time : np.ndarray or pd.Series
            Historical time array
        rate : np.ndarray or pd.Series
            Historical production rate
        n_periods : int, default 12
            Number of periods to forecast
        period_length : float, default 1.0
            Length of each period
        model_type : str, default 'hyperbolic'
            Decline model type
        n_samples : int, default 1000
            Number of Monte Carlo samples
        parameter_uncertainty : dict, optional
            Parameter uncertainty distributions (mean, std) for each parameter
            
        Returns
        -------
        dict
            Ensemble forecast results including:
            - 'forecasts': Array of individual forecasts
            - 'mean': Mean forecast
            - 'std': Standard deviation
            - 'quantiles': Quantile forecasts
        """
        from .decline_models import fit_decline_model
        
        # Fit base model
        base_model = fit_decline_model(time, rate, model_type=model_type)
        
        # Get base parameters
        base_params = base_model.params
        
        # Set default parameter uncertainty if not provided
        if parameter_uncertainty is None:
            parameter_uncertainty = self._get_default_uncertainty(
                model_type,
                base_params
            )
        
        # Generate forecast times
        last_time = np.max(time) if isinstance(time, np.ndarray) else time.max()
        forecast_times = np.arange(n_periods) * period_length + last_time + period_length
        
        # Generate ensemble
        forecasts = []
        
        for i in range(n_samples):
            # Sample parameters
            sampled_params = self._sample_parameters(
                base_params,
                parameter_uncertainty,
                model_type
            )
            
            # Create temporary model with sampled parameters
            temp_model = self._create_model_with_params(model_type, sampled_params)
            
            # Forecast
            forecast = temp_model.predict(forecast_times)
            forecasts.append(forecast)
        
        forecasts = np.array(forecasts)
        
        # Calculate statistics
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)
        
        # Calculate quantiles
        quantiles = {
            'p05': np.quantile(forecasts, 0.05, axis=0),
            'p25': np.quantile(forecasts, 0.25, axis=0),
            'p50': np.quantile(forecasts, 0.50, axis=0),
            'p75': np.quantile(forecasts, 0.75, axis=0),
            'p95': np.quantile(forecasts, 0.95, axis=0)
        }
        
        return {
            'forecasts': forecasts,
            'mean': mean_forecast,
            'std': std_forecast,
            'quantiles': quantiles,
            'forecast_times': forecast_times,
            'n_samples': n_samples
        }
    
    def _get_default_uncertainty(
        self,
        model_type: str,
        base_params: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Get default parameter uncertainty distributions."""
        uncertainty = {}
        
        for param_name, param_value in base_params.items():
            # Use 10% coefficient of variation as default
            std = abs(param_value) * 0.10
            uncertainty[param_name] = (param_value, std)
        
        return uncertainty
    
    def _sample_parameters(
        self,
        base_params: Dict[str, float],
        uncertainty: Dict[str, Tuple[float, float]],
        model_type: str
    ) -> Dict[str, float]:
        """Sample parameters from uncertainty distributions."""
        sampled = {}
        
        for param_name in base_params.keys():
            if param_name in uncertainty:
                mean, std = uncertainty[param_name]
                # Use lognormal for positive parameters
                if mean > 0:
                    # Convert to lognormal parameters
                    log_mean = np.log(mean)
                    log_std = std / mean  # Approximate
                    sampled_value = np.random.lognormal(log_mean, log_std)
                else:
                    sampled_value = np.random.normal(mean, std)
            else:
                sampled_value = base_params[param_name]
            
            # Ensure physical constraints
            if param_name in ['qi', 'di']:
                sampled_value = max(0.0, sampled_value)
            elif param_name == 'b':
                sampled_value = np.clip(sampled_value, 0.01, 0.99)
            
            sampled[param_name] = sampled_value
        
        return sampled
    
    def _create_model_with_params(
        self,
        model_type: str,
        params: Dict[str, float]
    ):
        """Create model instance with specified parameters."""
        from .decline_models import ExponentialDecline, HyperbolicDecline, HarmonicDecline
        
        model_map = {
            'exponential': ExponentialDecline,
            'hyperbolic': HyperbolicDecline,
            'harmonic': HarmonicDecline
        }
        
        model = model_map[model_type]()
        model.params = params
        model.fitted = True
        
        return model


def ensemble_forecast(
    time: Union[np.ndarray, pd.Series],
    rate: Union[np.ndarray, pd.Series],
    n_periods: int = 12,
    n_samples: int = 1000,
    model_type: str = 'hyperbolic'
) -> Dict[str, Any]:
    """
    Convenience function for ensemble forecasting.
    
    Parameters
    ----------
    time : np.ndarray or pd.Series
        Historical time
    rate : np.ndarray or pd.Series
        Historical production rate
    n_periods : int, default 12
        Number of periods to forecast
    n_samples : int, default 1000
        Number of Monte Carlo samples
    model_type : str, default 'hyperbolic'
        Decline model type
        
    Returns
    -------
    dict
        Ensemble forecast results
    """
    forecaster = MonteCarloForecaster()
    return forecaster.forecast_ensemble(
        time=time,
        rate=rate,
        n_periods=n_periods,
        n_samples=n_samples,
        model_type=model_type
    )


def forecast_uncertainty_bands(
    ensemble_result: Dict[str, Any],
    quantiles: Tuple[float, float] = (0.05, 0.95)
) -> pd.DataFrame:
    """
    Extract uncertainty bands from ensemble forecast.
    
    Parameters
    ----------
    ensemble_result : dict
        Result from ensemble_forecast
    quantiles : tuple, default (0.05, 0.95)
        Quantiles for uncertainty bands
        
    Returns
    -------
    pd.DataFrame
        Forecast with uncertainty bands
    """
    lower = np.quantile(ensemble_result['forecasts'], quantiles[0], axis=0)
    upper = np.quantile(ensemble_result['forecasts'], quantiles[1], axis=0)
    
    return pd.DataFrame({
        'time': ensemble_result['forecast_times'],
        'rate_mean': ensemble_result['mean'],
        'rate_std': ensemble_result['std'],
        'rate_lower': lower,
        'rate_upper': upper,
        'rate_p50': ensemble_result['quantiles']['p50']
    })


