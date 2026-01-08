"""
Bayesian posterior sampling for decline curves.

Provides Bayesian analysis of decline curve parameters with uncertainty
quantification and posterior sampling.
"""
import logging
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Bayesian libraries
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning(
        "PyMC not available. Bayesian decline analysis requires PyMC. "
        "Install with: pip install pymc arviz"
    )

try:
    from scipy.stats import norm, uniform, lognorm
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False


class BayesianDeclineAnalyzer:
    """
    Bayesian analyzer for decline curve parameters.
    
    Uses MCMC sampling to estimate posterior distributions of decline
    curve parameters, providing uncertainty quantification.
    
    Example:
        >>> analyzer = BayesianDeclineAnalyzer()
        >>> samples = analyzer.fit_and_sample(time, rate, model_type='hyperbolic')
        >>> forecast = analyzer.forecast_with_uncertainty(samples, n_periods=12)
    """
    
    def __init__(
        self,
        n_samples: int = 2000,
        n_tune: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize Bayesian decline analyzer.
        
        Parameters
        ----------
        n_samples : int, default 2000
            Number of MCMC samples
        n_tune : int, default 1000
            Number of tuning samples
        random_state : int, optional
            Random state for reproducibility
        """
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC is required for Bayesian decline analysis. "
                "Install with: pip install pymc arviz"
            )
        
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.random_state = random_state
        self.trace = None
        self.model = None
    
    def fit_and_sample(
        self,
        time: Union[np.ndarray, pd.Series],
        rate: Union[np.ndarray, pd.Series],
        model_type: str = 'hyperbolic',
        prior_params: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Any:
        """
        Fit decline model and sample from posterior distribution.
        
        Parameters
        ----------
        time : np.ndarray or pd.Series
            Time array
        rate : np.ndarray or pd.Series
            Production rate
        model_type : str, default 'hyperbolic'
            Type of decline model
        prior_params : dict, optional
            Prior parameter distributions
            
        Returns
        -------
        InferenceData
            MCMC trace with posterior samples
        """
        time_arr = np.asarray(time, dtype=float)
        rate_arr = np.asarray(rate, dtype=float)
        
        # Remove invalid values
        valid_mask = (rate_arr > 0) & np.isfinite(rate_arr) & np.isfinite(time_arr)
        time_arr = time_arr[valid_mask]
        rate_arr = rate_arr[valid_mask]
        
        if len(time_arr) < 3:
            raise ValueError("Need at least 3 valid data points")
        
        # Normalize time
        t0 = time_arr[0]
        time_normalized = time_arr - t0
        
        # Set default priors if not provided
        if prior_params is None:
            prior_params = self._get_default_priors(model_type, rate_arr)
        
        # Build PyMC model
        with pm.Model() as model:
            # Define priors
            if model_type == 'exponential':
                qi = pm.Lognormal('qi', mu=np.log(rate_arr[0]), sigma=0.5)
                di = pm.Lognormal('di', mu=np.log(0.01), sigma=1.0)
                
                # Likelihood
                rate_pred = qi * pm.math.exp(-di * time_normalized)
                pm.Lognormal('obs', mu=pm.math.log(rate_pred), sigma=0.1, observed=rate_arr)
                
            elif model_type == 'hyperbolic':
                qi = pm.Lognormal('qi', mu=np.log(rate_arr[0]), sigma=0.5)
                di = pm.Lognormal('di', mu=np.log(0.01), sigma=1.0)
                b = pm.Beta('b', alpha=2, beta=2)  # Prior centered at 0.5
                
                # Likelihood
                rate_pred = qi / pm.math.power(1 + b * di * time_normalized, 1.0 / b)
                pm.Lognormal('obs', mu=pm.math.log(rate_pred), sigma=0.1, observed=rate_arr)
                
            elif model_type == 'harmonic':
                qi = pm.Lognormal('qi', mu=np.log(rate_arr[0]), sigma=0.5)
                di = pm.Lognormal('di', mu=np.log(0.01), sigma=1.0)
                
                # Likelihood
                rate_pred = qi / (1 + di * time_normalized)
                pm.Lognormal('obs', mu=pm.math.log(rate_pred), sigma=0.1, observed=rate_arr)
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Sample
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                random_seed=self.random_state,
                return_inferencedata=True
            )
        
        self.model = model
        logger.info(f"Completed Bayesian sampling: {len(self.trace.posterior.draw)} samples")
        
        return self.trace
    
    def _get_default_priors(
        self,
        model_type: str,
        rate: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Get default prior parameters."""
        return {
            'qi': {'mu': np.log(rate[0]), 'sigma': 0.5},
            'di': {'mu': np.log(0.01), 'sigma': 1.0}
        }
    
    def get_posterior_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of posterior distributions.
        
        Returns
        -------
        pd.DataFrame
            Posterior summary with mean, std, and credible intervals
        """
        if self.trace is None:
            raise ValueError("Must run fit_and_sample first")
        
        return az.summary(self.trace)
    
    def forecast_with_uncertainty(
        self,
        trace: Optional[Any] = None,
        forecast_times: Optional[np.ndarray] = None,
        n_periods: int = 12,
        period_length: float = 1.0,
        model_type: str = 'hyperbolic',
        quantiles: Tuple[float, float] = (0.05, 0.95)
    ) -> pd.DataFrame:
        """
        Forecast production with uncertainty from posterior samples.
        
        Parameters
        ----------
        trace : InferenceData, optional
            MCMC trace (uses self.trace if not provided)
        forecast_times : np.ndarray, optional
            Specific times to forecast
        n_periods : int, default 12
            Number of periods to forecast
        period_length : float, default 1.0
            Length of each period
        model_type : str, default 'hyperbolic'
            Decline model type
        quantiles : tuple, default (0.05, 0.95)
            Quantiles for uncertainty bands
            
        Returns
        -------
        pd.DataFrame
            Forecast with uncertainty bands
        """
        if trace is None:
            trace = self.trace
        
        if trace is None:
            raise ValueError("Must provide trace or run fit_and_sample first")
        
        if forecast_times is None:
            forecast_times = np.arange(n_periods) * period_length
        
        # Get posterior samples
        posterior = trace.posterior
        
        # Sample from posterior
        # Get parameter arrays
        if model_type == 'exponential':
            qi_samples = posterior.qi.values.flatten()
            di_samples = posterior.di.values.flatten()
            n_samples = len(qi_samples)
            
            forecasts = []
            for i in range(min(100, n_samples)):  # Limit to 100 samples for speed
                qi = float(qi_samples[i])
                di = float(di_samples[i])
                rate_pred = qi * np.exp(-di * forecast_times)
                forecasts.append(rate_pred)
                
        elif model_type == 'hyperbolic':
            qi_samples = posterior.qi.values.flatten()
            di_samples = posterior.di.values.flatten()
            b_samples = posterior.b.values.flatten()
            n_samples = len(qi_samples)
            
            forecasts = []
            for i in range(min(100, n_samples)):  # Limit to 100 samples for speed
                qi = float(qi_samples[i])
                di = float(di_samples[i])
                b = float(b_samples[i])
                rate_pred = qi / np.power(1 + b * di * forecast_times, 1.0 / b)
                forecasts.append(rate_pred)
                
        elif model_type == 'harmonic':
            qi_samples = posterior.qi.values.flatten()
            di_samples = posterior.di.values.flatten()
            n_samples = len(qi_samples)
            
            forecasts = []
            for i in range(min(100, n_samples)):  # Limit to 100 samples for speed
                qi = float(qi_samples[i])
                di = float(di_samples[i])
                rate_pred = qi / (1 + di * forecast_times)
                forecasts.append(rate_pred)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        forecasts = np.array(forecasts)
        
        # Calculate statistics
        mean_forecast = np.mean(forecasts, axis=0)
        lower = np.quantile(forecasts, quantiles[0], axis=0)
        upper = np.quantile(forecasts, quantiles[1], axis=0)
        
        return pd.DataFrame({
            'time': forecast_times,
            'rate_mean': mean_forecast,
            'rate_lower': lower,
            'rate_upper': upper
        })


def sample_decline_posterior(
    time: Union[np.ndarray, pd.Series],
    rate: Union[np.ndarray, pd.Series],
    model_type: str = 'hyperbolic',
    n_samples: int = 2000
) -> Any:
    """
    Convenience function to sample from decline curve posterior.
    
    Parameters
    ----------
    time : np.ndarray or pd.Series
        Time array
    rate : np.ndarray or pd.Series
        Production rate
    model_type : str, default 'hyperbolic'
        Decline model type
    n_samples : int, default 2000
        Number of MCMC samples
        
    Returns
    -------
    InferenceData
        MCMC trace
    """
    analyzer = BayesianDeclineAnalyzer(n_samples=n_samples)
    return analyzer.fit_and_sample(time, rate, model_type=model_type)


def forecast_with_uncertainty(
    trace: Any,
    n_periods: int = 12,
    period_length: float = 1.0,
    model_type: str = 'hyperbolic'
) -> pd.DataFrame:
    """
    Convenience function to forecast with uncertainty.
    
    Parameters
    ----------
    trace : InferenceData
        MCMC trace from Bayesian sampling
    n_periods : int, default 12
        Number of periods to forecast
    period_length : float, default 1.0
        Length of each period
    model_type : str, default 'hyperbolic'
        Decline model type
        
    Returns
    -------
    pd.DataFrame
        Forecast with uncertainty bands
    """
    analyzer = BayesianDeclineAnalyzer()
    return analyzer.forecast_with_uncertainty(
        trace=trace,
        n_periods=n_periods,
        period_length=period_length,
        model_type=model_type
    )

