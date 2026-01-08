"""
Uncertainty quantification utilities for GeoSuite.

Provides error propagation, confidence intervals, and Monte Carlo
uncertainty analysis for derived petrophysical and geomechanical quantities.
"""
import logging
from typing import Union, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def propagate_error(
    func,
    *args,
    errors: Optional[Tuple[float, ...]] = None,
    covariance: Optional[np.ndarray] = None,
    method: str = 'first_order',
    n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainty through a function using error propagation.
    
    Uses first-order Taylor expansion (linear error propagation) or
    Monte Carlo sampling for nonlinear functions.
    
    Parameters
    ----------
    func : callable
        Function to propagate errors through
    *args
        Input arguments to the function
    errors : Tuple[float, ...], optional
        Standard errors for each input argument. Must match length of args.
    covariance : np.ndarray, optional
        Covariance matrix between inputs. If provided, accounts for correlations.
    method : str, default 'first_order'
        Method for error propagation: 'first_order' or 'monte_carlo'
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (result, uncertainty) where result is the function output and
        uncertainty is the propagated standard error
        
    Example
    -------
    >>> def porosity(rhob, rho_matrix=2.65, rho_fluid=1.0):
    ...     return (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    >>> 
    >>> rhob = np.array([2.3, 2.4, 2.5])
    >>> rhob_error = 0.05  # g/cc
    >>> result, uncertainty = propagate_error(porosity, rhob, errors=(rhob_error,))
    """
    if method == 'first_order':
        return _propagate_error_first_order(func, *args, errors=errors, covariance=covariance)
    elif method == 'monte_carlo':
        return _propagate_error_monte_carlo(func, *args, errors=errors, covariance=covariance, n_samples=n_samples)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'first_order' or 'monte_carlo'")


def _propagate_error_first_order(
    func,
    *args,
    errors: Optional[Tuple[float, ...]] = None,
    covariance: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order Taylor expansion error propagation."""
    # Calculate base result
    result = func(*args)
    
    if errors is None:
        return result, np.zeros_like(result)
    
    if len(errors) != len(args):
        raise ValueError(f"Number of errors ({len(errors)}) must match number of arguments ({len(args)})")
    
    # Calculate partial derivatives numerically
    h = 1e-6
    partials = []
    
    for i, (arg, error) in enumerate(zip(args, errors)):
        if isinstance(arg, (int, float)):
            # Scalar argument
            args_perturbed = list(args)
            args_perturbed[i] = arg + h
            result_perturbed = func(*args_perturbed)
            partial = (result_perturbed - result) / h
            partials.append(partial)
        else:
            # Array argument
            args_perturbed = list(args)
            arg_perturbed = arg.copy() if isinstance(arg, np.ndarray) else arg.copy()
            if isinstance(arg_perturbed, pd.Series):
                arg_perturbed = arg_perturbed.values
            arg_perturbed = arg_perturbed.astype(float) + h
            args_perturbed[i] = arg_perturbed
            result_perturbed = func(*args_perturbed)
            partial = (result_perturbed - result) / h
            partials.append(partial)
    
    # Calculate uncertainty
    if covariance is not None:
        # Account for correlations
        partials_array = np.array(partials)
        uncertainty_sq = np.sum(partials_array * np.dot(covariance, partials_array), axis=0)
        uncertainty = np.sqrt(uncertainty_sq)
    else:
        # Independent errors
        uncertainty_sq = sum(partial ** 2 * err ** 2 for partial, err in zip(partials, errors))
        if isinstance(uncertainty_sq, np.ndarray):
            uncertainty = np.sqrt(uncertainty_sq)
        else:
            uncertainty = np.sqrt(uncertainty_sq)
    
    return result, uncertainty


def _propagate_error_monte_carlo(
    func,
    *args,
    errors: Optional[Tuple[float, ...]] = None,
    covariance: Optional[np.ndarray] = None,
    n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo error propagation."""
    # Calculate base result
    result = func(*args)
    
    if errors is None:
        return result, np.zeros_like(result)
    
    # Generate samples
    samples = []
    for arg, error in zip(args, errors):
        if isinstance(arg, (int, float)):
            # Scalar - sample from normal distribution
            samples.append(np.random.normal(arg, error, n_samples))
        else:
            # Array - sample for each element
            if isinstance(arg, pd.Series):
                arg = arg.values
            arg = np.asarray(arg)
            
            # Sample each element independently
            if arg.ndim == 0:
                # Scalar array
                samples.append(np.random.normal(arg, error, n_samples))
            else:
                # Array - sample independently for each element
                samples_array = np.array([
                    np.random.normal(arg, error, n_samples)
                ]).T if arg.size == 1 else np.array([
                    np.random.normal(val, error, n_samples) for val in arg
                ]).T
                samples.append(samples_array)
    
    # Evaluate function on samples
    if len(args) == 1:
        # Single argument
        if samples[0].ndim == 1:
            result_samples = np.array([func(samples[0][i]) for i in range(n_samples)])
        else:
            result_samples = np.array([func(samples[0][i, :]) for i in range(n_samples)])
    else:
        # Multiple arguments - handle scalar and array cases
        result_samples = []
        for i in range(n_samples):
            sample_args = []
            for s in samples:
                if s.ndim == 1:
                    sample_args.append(s[i])
                else:
                    sample_args.append(s[i, :])
            result_samples.append(func(*sample_args))
        result_samples = np.array(result_samples)
    
    # Calculate statistics
    if result_samples.ndim == 1:
        uncertainty = np.std(result_samples, axis=0)
    else:
        uncertainty = np.std(result_samples, axis=0)
    
    return result, uncertainty


def confidence_interval(
    values: Union[np.ndarray, pd.Series],
    uncertainty: Union[np.ndarray, pd.Series],
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for values with uncertainty.
    
    Parameters
    ----------
    values : np.ndarray or pd.Series
        Mean values
    uncertainty : np.ndarray or pd.Series
        Standard errors
    confidence : float, default 0.95
        Confidence level (0-1)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (lower_bound, upper_bound) confidence intervals
    """
    if isinstance(values, pd.Series):
        values = values.values
    if isinstance(uncertainty, pd.Series):
        uncertainty = uncertainty.values
    
    # Calculate z-score for confidence level
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    lower = values - z_score * uncertainty
    upper = values + z_score * uncertainty
    
    return lower, upper


def monte_carlo_uncertainty(
    func,
    *args,
    distributions: Tuple[str, ...],
    distribution_params: Tuple[Dict, ...],
    n_samples: int = 10000,
    return_samples: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform Monte Carlo uncertainty analysis.
    
    Parameters
    ----------
    func : callable
        Function to analyze
    *args
        Mean values for input arguments
    distributions : Tuple[str, ...]
        Distribution types for each argument ('normal', 'uniform', 'lognormal', etc.)
    distribution_params : Tuple[Dict, ...]
        Parameters for each distribution
    n_samples : int, default 10000
        Number of Monte Carlo samples
    return_samples : bool, default False
        If True, also return the sample array
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean, std) or (mean, std, samples) if return_samples=True
    """
    # Generate samples
    samples_list = []
    for arg, dist_type, dist_params in zip(args, distributions, distribution_params):
        if dist_type == 'normal':
            samples = np.random.normal(arg, dist_params.get('scale', 1.0), n_samples)
        elif dist_type == 'uniform':
            samples = np.random.uniform(
                dist_params.get('low', arg - 1),
                dist_params.get('high', arg + 1),
                n_samples
            )
        elif dist_type == 'lognormal':
            samples = np.random.lognormal(
                np.log(arg),
                dist_params.get('scale', 0.1),
                n_samples
            )
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        samples_list.append(samples)
    
    # Evaluate function on samples
    if len(args) == 1:
        result_samples = np.array([func(samples_list[0][i]) for i in range(n_samples)])
    else:
        result_samples = np.array([func(*[s[i] for s in samples_list]) for i in range(n_samples)])
    
    # Calculate statistics
    mean = np.mean(result_samples, axis=0)
    std = np.std(result_samples, axis=0)
    
    if return_samples:
        return mean, std, result_samples
    else:
        return mean, std

