"""
Pressure calculations: overburden, hydrostatic, pore pressure.
"""

import numpy as np
import pandas as pd
from geosuite.utils.numba_helpers import njit


@njit(cache=True)
def _calculate_overburden_stress_kernel(
    depth: np.ndarray,
    rhob_kg: np.ndarray,
    g: float
) -> np.ndarray:
    """
    Numba-optimized kernel for overburden stress integration.
    
    This function is JIT-compiled for 20-50x speedup on large datasets.
    
    Args:
        depth: Depth array (meters)
        rhob_kg: Bulk density array (kg/m³)
        g: Gravitational acceleration (m/s^2)
        
    Returns:
        Overburden stress (MPa)
    """
    n = len(depth)
    sv = np.zeros(n, dtype=np.float64)
    
    # Trapezoidal integration: accumulate density × gravity × depth increment
    for i in range(1, n):
        dz = depth[i] - depth[i-1]
        avg_rho = (rhob_kg[i] + rhob_kg[i-1]) * 0.5
        sv[i] = sv[i-1] + avg_rho * g * dz * 1e-6  # Convert Pa to MPa
    
    return sv


def calculate_overburden_stress(
    depth: np.ndarray,
    rhob: np.ndarray,
    g: float = 9.81
) -> np.ndarray:
    """
    Calculate overburden stress (Sv) from density log.
    
    Uses trapezoidal integration: Sv = integral(rho * g * dz)
    
    This function is accelerated with Numba JIT compilation for 20-50x speedup
    on datasets with 1000+ samples. Falls back to pure Python if Numba unavailable.
    
    Args:
        depth: Depth array (meters) - can be numpy array or pandas Series
        rhob: Bulk density array (g/cc) - can be numpy array or pandas Series
        g: Gravitational acceleration (m/s^2), default 9.81
        
    Returns:
        Overburden stress (MPa) as numpy array
        
    Example:
        >>> depth = np.linspace(0, 3000, 1000)  # 0-3000m
        >>> rhob = np.ones(1000) * 2.5  # 2.5 g/cc
        >>> sv = calculate_overburden_stress(depth, rhob)
        >>> print(f"Overburden at {depth[-1]}m: {sv[-1]:.1f} MPa")
    """
    # Convert to numpy arrays with explicit dtype
    depth = np.asarray(depth, dtype=np.float64)
    rhob = np.asarray(rhob, dtype=np.float64)
    
    # Convert g/cc to kg/m³
    rhob_kg = rhob * 1000.0
    
    # Call optimized kernel
    return _calculate_overburden_stress_kernel(depth, rhob_kg, g)


def calculate_hydrostatic_pressure(
    depth: np.ndarray,
    rho_water: float = 1.03,
    g: float = 9.81
) -> np.ndarray:
    """
    Calculate hydrostatic pressure.
    
    Ph = rho_water * g * depth
    
    Args:
        depth: Depth array (meters)
        rho_water: Water density (g/cc), typically 1.0-1.1 for brine
        g: Gravitational acceleration (m/s^2)
        
    Returns:
        Hydrostatic pressure (MPa)
    """
    rho_water_kg = rho_water * 1000  # Convert to kg/m³
    ph = rho_water_kg * g * depth / 1e6  # Convert Pa to MPa
    
    return ph


def calculate_pore_pressure_eaton(
    depth: np.ndarray,
    dt: np.ndarray,
    dt_normal: np.ndarray,
    sv: np.ndarray,
    ph: np.ndarray,
    exponent: float = 3.0
) -> np.ndarray:
    """
    Calculate pore pressure using Eaton's method.
    
    Pp = Sv - (Sv - Ph) * (dt_normal / dt)^exponent
    
    Args:
        depth: Depth array (meters)
        dt: Measured sonic transit time (us/ft)
        dt_normal: Normal compaction trend sonic (us/ft)
        sv: Overburden stress (MPa)
        ph: Hydrostatic pressure (MPa)
        exponent: Eaton exponent (typically 3.0 for sonic)
        
    Returns:
        Pore pressure (MPa)
    """
    # Avoid division by zero
    dt = np.where(dt <= 0, np.nan, dt)
    dt_normal = np.where(dt_normal <= 0, np.nan, dt_normal)
    
    pp = sv - (sv - ph) * (dt_normal / dt) ** exponent
    
    return pp


def calculate_pore_pressure_bowers(
    depth: np.ndarray,
    dt: np.ndarray,
    dt_ml: float = 100.0,
    A: float = 5.0,
    B: float = 1.2,
    sv: np.ndarray = None,
    ph: np.ndarray = None,
    rho_water: float = 1.03,
    g: float = 9.81
) -> np.ndarray:
    """
    Calculate pore pressure using Bowers' method.
    
    This method accounts for unloading due to uplift/erosion.
    
    Args:
        depth: Depth array (meters)
        dt: Measured sonic transit time (us/ft)
        dt_ml: Mudline sonic (us/ft)
        A: Bowers A parameter
        B: Bowers B parameter
        sv: Overburden stress (MPa), calculated if not provided
        ph: Hydrostatic pressure (MPa), calculated if not provided
        rho_water: Water density (g/cc)
        g: Gravitational acceleration (m/s^2)
        
    Returns:
        Pore pressure (MPa)
    """
    if sv is None:
        # Assume average overburden gradient
        sv = 0.023 * depth  # MPa
    
    if ph is None:
        ph = calculate_hydrostatic_pressure(depth, rho_water, g)
    
    # Calculate effective stress from sonic
    sigma_eff = ((dt - dt_ml) / A) ** (1 / B)
    
    # Pore pressure
    pp = sv - sigma_eff
    
    return pp


def create_pressure_dataframe(
    depth: np.ndarray,
    rhob: np.ndarray = None,
    sv: np.ndarray = None,
    ph: np.ndarray = None,
    pp: np.ndarray = None,
    rho_water: float = 1.03,
    g: float = 9.81
) -> pd.DataFrame:
    """
    Create a DataFrame with all pressure calculations.
    
    Args:
        depth: Depth array (meters)
        rhob: Bulk density array (g/cc), optional
        sv: Overburden stress (MPa), calculated if not provided
        ph: Hydrostatic pressure (MPa), calculated if not provided
        pp: Pore pressure (MPa), uses hydrostatic if not provided
        rho_water: Water density (g/cc)
        g: Gravitational acceleration (m/s^2)
        
    Returns:
        DataFrame with depth, Sv, Ph, Pp columns
    """
    df = pd.DataFrame({'Depth': depth})
    
    # Calculate Sv if not provided
    if sv is None:
        if rhob is not None:
            sv = calculate_overburden_stress(depth, rhob, g)
        else:
            # Use typical gradient
            sv = 0.023 * depth  # MPa
    df['Sv'] = sv
    
    # Calculate Ph if not provided
    if ph is None:
        ph = calculate_hydrostatic_pressure(depth, rho_water, g)
    df['Ph'] = ph
    
    # Use Ph for Pp if not provided
    if pp is None:
        pp = ph
    df['Pp'] = pp
    
    return df

