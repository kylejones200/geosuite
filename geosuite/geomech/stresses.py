"""
Stress calculations: effective stress, overpressure, stress ratios.
"""

import numpy as np
import pandas as pd
from typing import Union
from geosuite.utils.numba_helpers import njit


def calculate_effective_stress(
    sv: Union[np.ndarray, pd.Series],
    pp: Union[np.ndarray, pd.Series],
    biot: float = 1.0
) -> np.ndarray:
    """
    Calculate vertical effective stress.
    
    σ'v = Sv - α * Pp
    
    Args:
        sv: Overburden stress (MPa)
        pp: Pore pressure (MPa)
        biot: Biot coefficient (typically 0.7-1.0)
        
    Returns:
        Effective stress (MPa)
    """
    return sv - biot * pp


def calculate_overpressure(
    pp: Union[np.ndarray, pd.Series],
    ph: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Calculate overpressure.
    
    ΔP = Pp - Ph
    
    Args:
        pp: Pore pressure (MPa)
        ph: Hydrostatic pressure (MPa)
        
    Returns:
        Overpressure (MPa)
    """
    return pp - ph


@njit(cache=True)
def _calculate_pressure_gradient_kernel(pressure: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Numba-optimized kernel for pressure gradient calculation.
    
    This function is JIT-compiled for 2-5x speedup.
    
    Args:
        pressure: Pressure array (MPa)
        depth: Depth array (meters)
        
    Returns:
        Pressure gradient (MPa/m)
    """
    n = len(pressure)
    gradient = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        dz = depth[i] - depth[i-1]
        if dz > 0.0:
            gradient[i] = (pressure[i] - pressure[i-1]) / dz
        else:
            gradient[i] = gradient[i-1] if i > 1 else 0.0
    
    # Extrapolate first value
    gradient[0] = gradient[1] if n > 1 else 0.0
    
    return gradient


def calculate_pressure_gradient(
    pressure: Union[np.ndarray, pd.Series],
    depth: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Calculate pressure gradient (MPa/m or equivalent mud weight).
    
    **Performance:** Accelerated with Numba JIT compilation for 2-5x speedup.
    Falls back to pure Python if Numba unavailable.
    
    Args:
        pressure: Pressure array (MPa)
        depth: Depth array (meters)
        
    Returns:
        Pressure gradient (MPa/m) as numpy array
    """
    pressure = np.asarray(pressure, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    
    if len(pressure) == 0 or len(depth) == 0:
        raise ValueError("Pressure and depth arrays must not be empty")
    
    if len(pressure) != len(depth):
        raise ValueError("Pressure and depth arrays must have same length")
    
    # Call optimized kernel
    return _calculate_pressure_gradient_kernel(pressure, depth)


def pressure_to_mud_weight(
    pressure: Union[np.ndarray, pd.Series],
    depth: Union[np.ndarray, pd.Series],
    g: float = 9.81
) -> np.ndarray:
    """
    Convert pressure to equivalent mud weight.
    
    MW = Pressure / (g * depth)
    
    Args:
        pressure: Pressure (MPa)
        depth: Depth (meters)
        g: Gravitational acceleration (m/s²), default 9.81
        
    Returns:
        Mud weight (g/cc) as numpy array
    """
    pressure = np.asarray(pressure, dtype=float)
    depth = np.asarray(depth, dtype=float)
    
    if len(pressure) == 0 or len(depth) == 0:
        raise ValueError("Pressure and depth arrays must not be empty")
    
    if len(pressure) != len(depth):
        raise ValueError("Pressure and depth arrays must have same length")
    
    # Avoid division by zero
    depth = np.where(depth <= 0, np.nan, depth)
    
    # Convert MPa to Pa, calculate density in kg/m³, then convert to g/cc
    mw = (pressure * 1e6) / (g * depth) / 1000
    
    return mw


def calculate_stress_ratio(
    shmin: Union[np.ndarray, pd.Series],
    sv: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Calculate horizontal to vertical stress ratio.
    
    k = Shmin / Sv
    
    Args:
        shmin: Minimum horizontal stress (MPa)
        sv: Vertical stress (MPa)
        
    Returns:
        Stress ratio (dimensionless) as numpy array
    """
    shmin = np.asarray(shmin, dtype=float)
    sv = np.asarray(sv, dtype=float)
    
    if len(shmin) == 0 or len(sv) == 0:
        raise ValueError("Stress arrays must not be empty")
    
    if len(shmin) != len(sv):
        raise ValueError("Stress arrays must have same length")
    
    sv = np.where(sv <= 0, np.nan, sv)
    return shmin / sv


def estimate_shmin_from_poisson(
    sv: Union[np.ndarray, pd.Series],
    pp: Union[np.ndarray, pd.Series],
    nu: float = 0.25,
    biot: float = 1.0
) -> np.ndarray:
    """
    Estimate minimum horizontal stress from Poisson's ratio.
    
    Shmin = (ν / (1 - ν)) * (Sv - α*Pp) + α*Pp
    
    Args:
        sv: Vertical stress (MPa)
        pp: Pore pressure (MPa)
        nu: Poisson's ratio, default 0.25
        biot: Biot coefficient, default 1.0
        
    Returns:
        Minimum horizontal stress (MPa) as numpy array
    """
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)
    
    if len(sv) == 0 or len(pp) == 0:
        raise ValueError("Stress and pressure arrays must not be empty")
    
    if len(sv) != len(pp):
        raise ValueError("Stress and pressure arrays must have same length")
    
    sigma_v_eff = sv - biot * pp
    shmin = (nu / (1 - nu)) * sigma_v_eff + biot * pp
    
    return shmin

