"""
AVO (Amplitude Versus Offset) attribute calculations from well log data.

This module provides functions to calculate AVO attributes including intercept,
gradient, curvature, reflectivity, and fluid factor from P-wave velocity,
S-wave velocity, and density measurements.

References:
    - Shuey, R. T. (1985). A simplification of the Zoeppritz equations.
      Geophysics, 50(4), 609-614.
    - Fatti, J. L., Smith, G. C., Vail, P. J., Strauss, P. J., & Levitt, P. R. (1994).
      Detection of gas in sandstone reservoirs using AVO analysis: A 3-D seismic
      case history using the Geostack technique. Geophysics, 59(9), 1362-1376.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd
from ..utils.numba_helpers import njit


def calculate_velocities_from_slowness(
    dtc: Union[np.ndarray, pd.Series],
    dts: Union[np.ndarray, pd.Series],
    units: str = 'm/s'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate P-wave and S-wave velocities from slowness (dtc, dts).
    
    Converts slowness (μs/ft or μs/m) to velocity (m/s or ft/s).
    
    Args:
        dtc: P-wave slowness (compressional, typically in μs/ft or μs/m)
        dts: S-wave slowness (shear, typically in μs/ft or μs/m)
        units: Input units, 'm/s' (default) or 'ft/s'
            - 'm/s': Assumes input in μs/m, converts to m/s
            - 'ft/s': Assumes input in μs/ft, converts to ft/s
            
    Returns:
        Tuple of (VP, VS) arrays in m/s or ft/s
        
    Example:
        >>> dtc = np.array([100, 120, 140])  # μs/ft
        >>> dts = np.array([180, 200, 220])  # μs/ft
        >>> vp, vs = calculate_velocities_from_slowness(dtc, dts, units='ft/s')
    """
    dtc = np.asarray(dtc, dtype=np.float64)
    dts = np.asarray(dts, dtype=np.float64)
    
    if units == 'm/s':
        # Input in μs/m, convert to m/s: v = 1e6 / dt
        vp = 1e6 / dtc
        vs = 1e6 / dts
    elif units == 'ft/s':
        # Input in μs/ft, convert to ft/s, then to m/s
        vp_ft_s = 1e6 / dtc
        vs_ft_s = 1e6 / dts
        # Convert ft/s to m/s
        vp = vp_ft_s / 3.281
        vs = vs_ft_s / 3.281
    else:
        raise ValueError(f"Unknown units: {units}. Must be 'm/s' or 'ft/s'")
    
    # Handle division by zero
    vp = np.where(np.isinf(vp) | (dtc <= 0), np.nan, vp)
    vs = np.where(np.isinf(vs) | (dts <= 0), np.nan, vs)
    
    return vp, vs


def preprocess_avo_inputs(
    vp: Union[np.ndarray, pd.Series],
    vs: Union[np.ndarray, pd.Series],
    rho: Union[np.ndarray, pd.Series]
) -> dict[str, np.ndarray]:
    """
    Preprocess velocity and density data for AVO calculations.
    
    Calculates average values and differences between consecutive samples,
    which are required for AVO attribute calculations.
    
    Args:
        vp: P-wave velocity (m/s)
        vs: S-wave velocity (m/s)
        rho: Density (g/cc or kg/m³)
        
    Returns:
        Dictionary with keys:
            - 'VP_AVG': Average P-wave velocity between consecutive samples
            - 'VS_AVG': Average S-wave velocity between consecutive samples
            - 'RHO': Average density between consecutive samples
            - 'dVp': Difference in P-wave velocity between consecutive samples
            - 'dVs': Difference in S-wave velocity between consecutive samples
            - 'dRho': Difference in density between consecutive samples
            
    Example:
        >>> vp = np.array([3000, 3100, 3200])
        >>> vs = np.array([1500, 1550, 1600])
        >>> rho = np.array([2.3, 2.4, 2.5])
        >>> preprocessed = preprocess_avo_inputs(vp, vs, rho)
    """
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    
    n = len(vp)
    
    if len(vs) != n or len(rho) != n:
        raise ValueError("vp, vs, and rho must have the same length")
    
    if n < 2:
        raise ValueError("Input arrays must have at least 2 samples")
    
    # Calculate differences
    dvp = np.zeros(n, dtype=np.float64)
    dvs = np.zeros(n, dtype=np.float64)
    drho = np.zeros(n, dtype=np.float64)
    
    # Calculate averages
    vp_avg = np.zeros(n, dtype=np.float64)
    vs_avg = np.zeros(n, dtype=np.float64)
    rho_avg = np.zeros(n, dtype=np.float64)
    
    for i in range(n - 1):
        dvp[i] = vp[i + 1] - vp[i]
        dvs[i] = vs[i + 1] - vs[i]
        drho[i] = rho[i + 1] - rho[i]
        
        vp_avg[i] = (vp[i] + vp[i + 1]) / 2.0
        vs_avg[i] = (vs[i] + vs[i + 1]) / 2.0
        rho_avg[i] = (rho[i] + rho[i + 1]) / 2.0
    
    # Use last calculated values for final sample
    dvp[-1] = dvp[-2]
    dvs[-1] = dvs[-2]
    drho[-1] = drho[-2]
    vp_avg[-1] = vp_avg[-2]
    vs_avg[-1] = vs_avg[-2]
    rho_avg[-1] = rho_avg[-2]
    
    return {
        'VP_AVG': vp_avg,
        'VS_AVG': vs_avg,
        'RHO': rho_avg,
        'dVp': dvp,
        'dVs': dvs,
        'dRho': drho
    }


def calculate_avo_attributes(
    vp: Union[np.ndarray, pd.Series],
    vs: Union[np.ndarray, pd.Series],
    rho: Union[np.ndarray, pd.Series],
    return_all: bool = True
) -> pd.DataFrame | dict[str, np.ndarray]:
    """
    Calculate AVO attributes from well log velocities and density.
    
    Computes AVO attributes including:
    - Intercept (A): Zero-offset P-wave reflectivity
    - Gradient (B): AVO gradient
    - Curvature (C): Second-order AVO term
    - Poisson's Ratio (PR)
    - P-wave and S-wave reflectivity (Rp, Rs)
    - Fluid Factor (FF)
    - Product attributes (A*B, A*sign(B), B*sign(A))
    
    Based on the Shuey (1985) approximation of the Zoeppritz equations.
    
    Args:
        vp: P-wave velocity (m/s)
        vs: S-wave velocity (m/s)
        rho: Density (g/cc)
        return_all: If True, return DataFrame with all attributes.
                   If False, return dict with only key attributes.
        
    Returns:
        DataFrame or dict with AVO attributes. If DataFrame, includes columns:
            - 'k': Shear modulus ratio (VS/VP)^2
            - 'A': Intercept (zero-offset reflectivity)
            - 'B': Gradient
            - 'C': Curvature
            - 'productAB': A * B
            - 'AsignB': A * sign(B)
            - 'BsignA': B * sign(A)
            - 'PR': Poisson's Ratio
            - 'Rp': P-wave reflectivity
            - 'Rs': S-wave reflectivity
            - 'FF': Fluid Factor
            
    Example:
        >>> vp = np.array([3000, 3100, 3200])
        >>> vs = np.array([1500, 1550, 1600])
        >>> rho = np.array([2.3, 2.4, 2.5])
        >>> avo_df = calculate_avo_attributes(vp, vs, rho)
        >>> print(avo_df[['A', 'B', 'PR']].head())
    """
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    
    # Preprocess inputs
    preprocessed = preprocess_avo_inputs(vp, vs, rho)
    
    vp_avg = preprocessed['VP_AVG']
    vs_avg = preprocessed['VS_AVG']
    rho_avg = preprocessed['RHO']
    dvp = preprocessed['dVp']
    dvs = preprocessed['dVs']
    drho = preprocessed['dRho']
    
    # Shear modulus ratio: k = (VS/VP)^2
    k = (vs_avg / vp_avg) ** 2
    k = np.where(np.isnan(k) | np.isinf(k), 0, k)
    
    # Intercept (A): Zero-offset P-wave reflectivity
    # A = 0.5 * (dVp/VP_AVG + dRho/RHO)
    A = 0.5 * ((dvp / vp_avg) + (drho / rho_avg))
    
    # Gradient (B)
    # B = 0.5*(dVp/VP_AVG) - 2*k*(2*(dVs/VS_AVG) + dRho/RHO)
    B = 0.5 * (dvp / vp_avg) - (2 * k) * ((2 * (dvs / vs_avg)) + (drho / rho_avg))
    
    # Curvature (C)
    C = 0.5 * (dvp / vp_avg)
    
    # Product attributes
    productAB = A * B
    AsignB = A * np.sign(B)
    BsignA = B * np.sign(A)
    
    # Poisson's Ratio: PR = (gamma^2 - 2) / (2*gamma^2 - 2)
    # where gamma = VP/VS
    gamma = vp_avg / vs_avg
    gamma_sq = gamma ** 2
    PR = (gamma_sq - 2) / (2 * gamma_sq - 2)
    PR = np.where(np.isnan(PR) | np.isinf(PR), np.nan, PR)
    
    # P-wave Reflectivity (Rp)
    Rp = A.copy()
    
    # S-wave Reflectivity (Rs)
    Rs = 0.5 * (A - B)
    
    # Fluid Factor (Fatti et al., 1994)
    # FF = Rp - 1.16 * (VS_AVG/VP_AVG) * Rs
    vs_vp_ratio = vs_avg / vp_avg
    FF = Rp - (1.16 * vs_vp_ratio * Rs)
    
    # Create result dictionary
    result_dict = {
        'k': k,
        'A': A,
        'B': B,
        'C': C,
        'productAB': productAB,
        'AsignB': AsignB,
        'BsignA': BsignA,
        'PR': PR,
        'Rp': Rp,
        'Rs': Rs,
        'FF': FF
    }
    
    if return_all:
        return pd.DataFrame(result_dict)
    else:
        # Return only key attributes
        return {
            'A': A,
            'B': B,
            'PR': PR,
            'Rp': Rp,
            'Rs': Rs,
            'FF': FF
        }


def calculate_avo_from_slowness(
    dtc: Union[np.ndarray, pd.Series],
    dts: Union[np.ndarray, pd.Series],
    rho: Union[np.ndarray, pd.Series],
    units: str = 'ft/s',
    return_all: bool = True
) -> pd.DataFrame | dict[str, np.ndarray]:
    """
    Calculate AVO attributes directly from slowness (dtc, dts) and density.
    
    Convenience function that combines velocity calculation and AVO attribute
    calculation in one step.
    
    Args:
        dtc: P-wave slowness (typically μs/ft or μs/m)
        dts: S-wave slowness (typically μs/ft or μs/m)
        rho: Density (g/cc)
        units: Input units for slowness, 'ft/s' (default) or 'm/s'
        return_all: If True, return DataFrame with all attributes.
                   If False, return dict with only key attributes.
        
    Returns:
        DataFrame or dict with AVO attributes (see calculate_avo_attributes)
        
    Example:
        >>> dtc = np.array([100, 120, 140])  # μs/ft
        >>> dts = np.array([180, 200, 220])  # μs/ft
        >>> rho = np.array([2.3, 2.4, 2.5])  # g/cc
        >>> avo_df = calculate_avo_from_slowness(dtc, dts, rho, units='ft/s')
    """
    vp, vs = calculate_velocities_from_slowness(dtc, dts, units=units)
    return calculate_avo_attributes(vp, vs, rho, return_all=return_all)

