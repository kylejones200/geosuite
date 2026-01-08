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
            if arg.ndim == 0 or arg.size == 1:
                # Scalar array
                samples.append(np.random.normal(float(arg), error, n_samples))
            else:
                # Array - sample independently for each element
                arg_flat = arg.flatten()
                samples_array = np.array([
                    np.random.normal(val, error, n_samples) for val in arg_flat
                ]).T
                if arg.ndim > 1:
                    samples_array = samples_array.reshape((n_samples,) + arg.shape)
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
    
    # Ensure uncertainty matches result shape
    if isinstance(result, np.ndarray) and result.ndim > 0:
        uncertainty = np.reshape(uncertainty, result.shape)
    
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


def uncertainty_porosity_from_density(
    rhob: Union[np.ndarray, pd.Series],
    rhob_error: Union[float, np.ndarray, pd.Series],
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0,
    rho_matrix_error: float = 0.0,
    rho_fluid_error: float = 0.0,
    method: str = 'first_order',
    n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate porosity from density with uncertainty propagation.
    
    Args:
        rhob: Bulk density (g/cc)
        rhob_error: Uncertainty in bulk density (g/cc)
        rho_matrix: Matrix density (g/cc, default 2.65)
        rho_fluid: Fluid density (g/cc, default 1.0)
        rho_matrix_error: Uncertainty in matrix density (g/cc)
        rho_fluid_error: Uncertainty in fluid density (g/cc)
        method: Error propagation method ('first_order' or 'monte_carlo')
        
    Returns:
        Tuple of (porosity, porosity_uncertainty)
    """
    from geosuite.petro.calculations import calculate_porosity_from_density
    
    def porosity_func(rhob_val, rho_mat=rho_matrix, rho_fl=rho_fluid):
        return calculate_porosity_from_density(rhob_val, rho_mat, rho_fl)
    
    rhob = np.asarray(rhob)
    rhob_error = np.asarray(rhob_error) if not isinstance(rhob_error, (int, float)) else rhob_error
    
    if isinstance(rhob_error, (int, float)):
        errors = (rhob_error, rho_matrix_error, rho_fluid_error)
    else:
        errors = (rhob_error, np.full_like(rhob, rho_matrix_error), np.full_like(rhob, rho_fluid_error))
    
    return propagate_error(porosity_func, rhob, rho_matrix, rho_fluid, errors=errors, method=method, n_samples=n_samples)


def uncertainty_water_saturation(
    phi: Union[np.ndarray, pd.Series],
    rt: Union[np.ndarray, pd.Series],
    phi_error: Union[float, np.ndarray, pd.Series],
    rt_error: Union[float, np.ndarray, pd.Series],
    rw: float = 0.05,
    rw_error: float = 0.0,
    m: float = 2.0,
    m_error: float = 0.0,
    n: float = 2.0,
    n_error: float = 0.0,
    a: float = 1.0,
    a_error: float = 0.0,
    method: str = 'monte_carlo',
    n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate water saturation with uncertainty propagation.
    
    Args:
        phi: Porosity (fraction)
        rt: True resistivity (ohm-m)
        phi_error: Uncertainty in porosity
        rt_error: Uncertainty in resistivity (ohm-m)
        rw: Formation water resistivity (ohm-m)
        rw_error: Uncertainty in rw
        m: Cementation exponent
        m_error: Uncertainty in m
        n: Saturation exponent
        n_error: Uncertainty in n
        a: Tortuosity factor
        a_error: Uncertainty in a
        method: Error propagation method ('first_order' or 'monte_carlo')
        
    Returns:
        Tuple of (water_saturation, sw_uncertainty)
    """
    from geosuite.petro.calculations import calculate_water_saturation
    
    def sw_func(phi_val, rt_val, rw_val=rw, m_val=m, n_val=n, a_val=a):
        return calculate_water_saturation(phi_val, rt_val, rw_val, m_val, n_val, a_val)
    
    phi = np.asarray(phi)
    rt = np.asarray(rt)
    
    phi_error = np.asarray(phi_error) if not isinstance(phi_error, (int, float)) else phi_error
    rt_error = np.asarray(rt_error) if not isinstance(rt_error, (int, float)) else rt_error
    
    if isinstance(phi_error, (int, float)):
        errors = (phi_error, rt_error, rw_error, m_error, n_error, a_error)
    else:
        errors = (
            phi_error,
            rt_error,
            np.full_like(phi, rw_error),
            np.full_like(phi, m_error),
            np.full_like(phi, n_error),
            np.full_like(phi, a_error)
        )
    
    return propagate_error(sw_func, phi, rt, rw, m, n, a, errors=errors, method=method, n_samples=n_samples)


def uncertainty_permeability(
    phi: Union[np.ndarray, pd.Series],
    sw: Union[np.ndarray, pd.Series],
    phi_error: Union[float, np.ndarray, pd.Series],
    sw_error: Union[float, np.ndarray, pd.Series],
    method: str = 'timur',
    method_params: Optional[Dict[str, float]] = None,
    method_param_errors: Optional[Dict[str, float]] = None,
    uncertainty_method: str = 'monte_carlo',
    n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate permeability with uncertainty propagation.
    
    Args:
        phi: Porosity (fraction)
        sw: Water saturation (fraction)
        phi_error: Uncertainty in porosity
        sw_error: Uncertainty in water saturation
        method: Permeability model ('timur', 'wyllie_rose', 'coates_dumanoir')
        method_params: Model parameters (coefficient, exponents)
        method_param_errors: Uncertainties in model parameters
        uncertainty_method: Error propagation method ('first_order' or 'monte_carlo')
        
    Returns:
        Tuple of (permeability, permeability_uncertainty)
    """
    from geosuite.petro.permeability import (
        calculate_permeability_timur,
        calculate_permeability_wyllie_rose,
        calculate_permeability_coates_dumanoir,
    )
    
    method_params = method_params or {}
    method_param_errors = method_param_errors or {}
    
    model_configs = {
        'timur': (calculate_permeability_timur, {
            'coefficient': 0.136,
            'porosity_exponent': 4.4,
            'saturation_exponent': 2.0,
        }),
        'wyllie_rose': (calculate_permeability_wyllie_rose, {
            'coefficient': 0.625,
            'porosity_exponent': 6.0,
            'saturation_exponent': 2.0,
        }),
        'coates_dumanoir': (calculate_permeability_coates_dumanoir, {
            'coefficient': 70.0,
            'porosity_exponent': 2.0,
            'saturation_exponent': 2.0,
        }),
    }
    
    if method not in model_configs:
        raise ValueError(f"Unknown method: {method}. Choose: {', '.join(model_configs.keys())}")
    
    perm_func, default_params = model_configs[method]
    params = {**default_params, **method_params}
    
    def perm_calc(phi_val, sw_val, **kwargs):
        return perm_func(phi_val, sw_val, **kwargs)
    
    phi = np.asarray(phi)
    sw = np.asarray(sw)
    
    phi_error = np.asarray(phi_error) if not isinstance(phi_error, (int, float)) else phi_error
    sw_error = np.asarray(sw_error) if not isinstance(sw_error, (int, float)) else sw_error
    
    errors_list = [phi_error, sw_error]
    args_list = [phi, sw]
    
    for param_name in ['coefficient', 'porosity_exponent', 'saturation_exponent']:
        if param_name in params:
            args_list.append(params[param_name])
            param_error = method_param_errors.get(param_name, 0.0)
            if isinstance(phi_error, (int, float)):
                errors_list.append(param_error)
            else:
                errors_list.append(np.full_like(phi, param_error))
    
    def wrapped_func(*args):
        phi_val, sw_val = args[0], args[1]
        param_dict = {}
        if len(args) > 2:
            param_dict['coefficient'] = args[2] if len(args) > 2 else params['coefficient']
            param_dict['porosity_exponent'] = args[3] if len(args) > 3 else params['porosity_exponent']
            param_dict['saturation_exponent'] = args[4] if len(args) > 4 else params['saturation_exponent']
        else:
            param_dict = params.copy()
        
        phi_val = np.atleast_1d(np.asarray(phi_val, dtype=float))
        sw_val = np.atleast_1d(np.asarray(sw_val, dtype=float))
        
        result = perm_func(phi_val, sw_val, **param_dict)
        
        return result[0] if result.size == 1 else result
    
    return propagate_error(wrapped_func, *args_list, errors=tuple(errors_list), method=uncertainty_method, n_samples=n_samples)

