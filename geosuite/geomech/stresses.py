"""
Stress calculations: effective stress, overpressure, stress ratios.
"""

import numpy as np
from geosuite.utils.numba_helpers import njit


def calculate_effective_stress(
    sv: np.ndarray,
    pp: np.ndarray,
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
    pp: np.ndarray,
    ph: np.ndarray
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
    pressure: np.ndarray,
    depth: np.ndarray
) -> np.ndarray:
    """
    Calculate pressure gradient (MPa/m or equivalent mud weight).
    
    **Performance:** Accelerated with Numba JIT compilation for 2-5x speedup.
    Falls back to pure Python if Numba unavailable.
    
    Args:
        pressure: Pressure array (MPa) - can be numpy array or pandas Series
        depth: Depth array (meters) - can be numpy array or pandas Series
        
    Returns:
        Pressure gradient (MPa/m) as numpy array
    """
    # Convert to numpy arrays with explicit dtype
    pressure = np.asarray(pressure, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    
    # Call optimized kernel
    return _calculate_pressure_gradient_kernel(pressure, depth)


def pressure_to_mud_weight(
    pressure: np.ndarray,
    depth: np.ndarray,
    g: float = 9.81
) -> np.ndarray:
    """
    Convert pressure to equivalent mud weight.
    
    MW = Pressure / (g * depth)
    
    Args:
        pressure: Pressure (MPa)
        depth: Depth (meters)
        g: Gravitational acceleration (m/s^2)
        
    Returns:
        Mud weight (g/cc)
    """
    # Avoid division by zero
    depth = np.where(depth <= 0, np.nan, depth)
    
    # Convert MPa to Pa, calculate density in kg/m³, then convert to g/cc
    mw = (pressure * 1e6) / (g * depth) / 1000
    
    return mw


def calculate_stress_ratio(
    shmin: np.ndarray,
    sv: np.ndarray
) -> np.ndarray:
    """
    Calculate horizontal to vertical stress ratio.
    
    k = Shmin / Sv
    
    Args:
        shmin: Minimum horizontal stress (MPa)
        sv: Vertical stress (MPa)
        
    Returns:
        Stress ratio (dimensionless)
    """
    sv = np.where(sv <= 0, np.nan, sv)
    return shmin / sv


def estimate_shmin_from_poisson(
    sv: np.ndarray,
    pp: np.ndarray,
    nu: float = 0.25,
    biot: float = 1.0
) -> np.ndarray:
    """
    Estimate minimum horizontal stress from Poisson's ratio.
    
    Shmin = (ν / (1 - ν)) * (Sv - α*Pp) + α*Pp
    
    Args:
        sv: Vertical stress (MPa)
        pp: Pore pressure (MPa)
        nu: Poisson's ratio
        biot: Biot coefficient
        
    Returns:
        Minimum horizontal stress (MPa)
    """
    sigma_v_eff = sv - biot * pp
    shmin = (nu / (1 - nu)) * sigma_v_eff + biot * pp
    
    return shmin

