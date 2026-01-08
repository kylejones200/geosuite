"""
Rock physics transforms including Gassmann fluid substitution.

Provides tools for modeling the effects of fluid changes on seismic properties.
"""
import logging
from typing import Union, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def gassmann_fluid_substitution(
    k_sat_initial: Union[np.ndarray, pd.Series],
    k_dry: Union[np.ndarray, pd.Series],
    k_mineral: Union[float, np.ndarray, pd.Series],
    k_fluid_initial: Union[float, np.ndarray, pd.Series],
    k_fluid_final: Union[float, np.ndarray, pd.Series],
    phi: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Perform Gassmann fluid substitution to predict bulk modulus after fluid change.
    
    Gassmann's equation predicts how bulk modulus changes when pore fluid
    is replaced, assuming constant pore pressure and no chemical interactions.
    
    K_sat_final = K_dry + (1 - K_dry/K_mineral)^2 / (phi/K_fluid_final + (1-phi)/K_mineral - K_dry/K_mineral^2)
    
    Parameters
    ----------
    k_sat_initial : np.ndarray or pd.Series
        Initial saturated bulk modulus (GPa)
    k_dry : np.ndarray or pd.Series
        Dry frame bulk modulus (GPa)
    k_mineral : float, np.ndarray, or pd.Series
        Mineral bulk modulus (GPa). Typical values:
        - Quartz: 37 GPa
        - Calcite: 77 GPa
        - Dolomite: 95 GPa
    k_fluid_initial : float, np.ndarray, or pd.Series
        Initial fluid bulk modulus (GPa). Typical values:
        - Water: 2.2 GPa
        - Oil: 0.5-2.0 GPa
        - Gas: 0.01-0.1 GPa
    k_fluid_final : float, np.ndarray, or pd.Series
        Final fluid bulk modulus (GPa)
    phi : np.ndarray or pd.Series
        Porosity (fraction)
        
    Returns
    -------
    np.ndarray
        Final saturated bulk modulus (GPa)
        
    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or invalid values
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    k_sat_initial = np.asarray(k_sat_initial, dtype=float)
    k_dry = np.asarray(k_dry, dtype=float)
    
    # Broadcast scalars to arrays
    k_mineral = np.full_like(phi, k_mineral) if isinstance(k_mineral, (int, float)) else np.asarray(k_mineral, dtype=float)
    k_fluid_initial = np.full_like(phi, k_fluid_initial) if isinstance(k_fluid_initial, (int, float)) else np.asarray(k_fluid_initial, dtype=float)
    k_fluid_final = np.full_like(phi, k_fluid_final) if isinstance(k_fluid_final, (int, float)) else np.asarray(k_fluid_final, dtype=float)
    
    # Validate inputs
    lengths = [len(k_sat_initial), len(k_dry), len(k_mineral), len(k_fluid_initial), len(k_fluid_final), len(phi)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Got lengths: {lengths}"
        )
    
    validation_checks = {
        'porosity outside (0, 1)': np.any((phi <= 0) | (phi >= 1)),
        'non-positive bulk moduli': np.any((k_dry <= 0) | (k_mineral <= 0)),
    }
    
    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, results may be invalid")
    
    logger.debug(
        f"Performing Gassmann fluid substitution for {len(phi)} samples"
    )
    
    # Gassmann's equation
    # K_sat = K_dry + (1 - K_dry/K_mineral)^2 / (phi/K_fluid + (1-phi)/K_mineral - K_dry/K_mineral^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate denominator
        term1 = phi / k_fluid_final
        term2 = (1 - phi) / k_mineral
        term3 = k_dry / (k_mineral ** 2)
        denominator = term1 + term2 - term3
        
        # Avoid division by zero
        denominator = np.where(denominator <= 0, np.nan, denominator)
        
        # Calculate numerator
        numerator = (1 - k_dry / k_mineral) ** 2
        
        # Calculate final saturated bulk modulus
        k_sat_final = k_dry + numerator / denominator
    
    # Count NaN values
    nan_count = np.isnan(k_sat_final).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Gassmann calculation")
    
    return k_sat_final


def calculate_fluid_bulk_modulus(
    sw: Union[np.ndarray, pd.Series],
    so: Optional[Union[np.ndarray, pd.Series]] = None,
    sg: Optional[Union[np.ndarray, pd.Series]] = None,
    k_water: float = 2.2,
    k_oil: float = 1.0,
    k_gas: float = 0.05,
    temperature: float = 25.0,
    pressure: float = 20.0
) -> np.ndarray:
    """
    Calculate effective fluid bulk modulus from saturations.
    
    Uses Reuss average (isostress) for fluid mixing:
    K_fluid = 1 / (Sw/K_water + So/K_oil + Sg/K_gas)
    
    Parameters
    ----------
    sw : np.ndarray or pd.Series
        Water saturation (fraction)
    so : np.ndarray or pd.Series, optional
        Oil saturation (fraction). If None, calculated as 1 - Sw - Sg
    sg : np.ndarray or pd.Series, optional
        Gas saturation (fraction). If None, assumed to be 0
    k_water : float, default 2.2
        Water bulk modulus (GPa) at standard conditions
    k_oil : float, default 1.0
        Oil bulk modulus (GPa) at standard conditions
    k_gas : float, default 0.05
        Gas bulk modulus (GPa) at standard conditions
    temperature : float, default 25.0
        Temperature (°C) for pressure correction
    pressure : float, default 20.0
        Pressure (MPa) for pressure correction
        
    Returns
    -------
    np.ndarray
        Effective fluid bulk modulus (GPa)
    """
    sw = np.asarray(sw, dtype=float)
    
    # Calculate missing saturations
    sg = np.zeros_like(sw) if sg is None else np.asarray(sg, dtype=float)
    so = (1.0 - sw - sg) if so is None else np.asarray(so, dtype=float)
    
    # Validate and clip saturations
    saturation_bounds = np.any([(sw < 0) | (sw > 1), (so < 0) | (so > 1), (sg < 0) | (sg > 1)])
    if saturation_bounds:
        logger.warning("Found saturations outside [0, 1], clipping to valid range")
    
    sw, so, sg = np.clip(sw, 0, 1), np.clip(so, 0, 1), np.clip(sg, 0, 1)
    
    # Normalize to ensure Sw + So + Sg = 1
    total_sat = sw + so + sg
    sw = sw / total_sat
    so = so / total_sat
    sg = sg / total_sat
    
    # Apply pressure/temperature corrections (simplified)
    # Gas is most sensitive to pressure
    k_gas_corrected = k_gas * (1 + 0.01 * pressure)  # Rough correction
    
    # Reuss average (isostress)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_fluid = 1.0 / (sw / k_water + so / k_oil + sg / k_gas_corrected)
    
    # Handle invalid values
    k_fluid = np.where(np.isfinite(k_fluid), k_fluid, np.nan)
    
    logger.debug(f"Calculated fluid bulk modulus for {len(sw)} samples")
    
    return k_fluid


def calculate_density_from_velocity(
    vp: Union[np.ndarray, pd.Series],
    vs: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = "gardner"
) -> np.ndarray:
    """
    Estimate density from velocity using empirical relationships.
    
    Parameters
    ----------
    vp : np.ndarray or pd.Series
        P-wave velocity (m/s)
    vs : np.ndarray or pd.Series, optional
        S-wave velocity (m/s). If provided, uses more accurate method
    method : str, default "gardner"
        Method to use: "gardner", "nafe_drake", or "brocher"
        
    Returns
    -------
    np.ndarray
        Estimated density (g/cc)
    """
    vp = np.asarray(vp, dtype=float)
    vp_km_s = vp / 1000.0
    
    methods = {
        "gardner": lambda v: 1.74 * (v ** 0.25),
        "nafe_drake": lambda v: 1.5 + 0.5 * v,
        "brocher": lambda v: (
            1.6612 * v - 0.4721 * (v ** 2) + 0.0671 * (v ** 3) 
            - 0.0043 * (v ** 4) + 0.000106 * (v ** 5)
        ),
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose: {', '.join(methods.keys())}")
    
    rho = methods[method](vp_km_s)
    
    # Clip to reasonable range
    rho = np.clip(rho, 1.0, 3.5)
    
    logger.debug(f"Estimated density using {method} method for {len(vp)} samples")
    
    return rho

