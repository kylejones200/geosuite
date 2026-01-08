"""
Pressure calculations: overburden, hydrostatic, pore pressure.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, TYPE_CHECKING
from geosuite.utils.numba_helpers import njit

if TYPE_CHECKING:
    from ..config import ConfigManager
else:
    ConfigManager = None


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
    depth: Union[np.ndarray, pd.Series],
    rhob: Union[np.ndarray, pd.Series],
    g: Optional[float] = None,
    config: Optional['ConfigManager'] = None
) -> np.ndarray:
    """
    Calculate overburden stress (Sv) from density log.
    
    Uses trapezoidal integration: Sv = integral(rho * g * dz)
    
    This function is accelerated with Numba JIT compilation for 20-50x speedup
    on datasets with 1000+ samples. Falls back to pure Python if Numba unavailable.
    
    Args:
        depth: Depth array (meters)
        rhob: Bulk density array (g/cc)
        g: Gravitational acceleration (m/s²). If None, reads from config.
        config: ConfigManager instance. If None, uses global config.
        
    Returns:
        Overburden stress (MPa) as numpy array
        
    Example:
        >>> depth = np.linspace(0, 3000, 1000)  # 0-3000m
        >>> rhob = np.ones(1000) * 2.5  # 2.5 g/cc
        >>> sv = calculate_overburden_stress(depth, rhob)
        >>> print(f"Overburden at {depth[-1]}m: {sv[-1]:.1f} MPa")
    """
    # Load from config if not provided
    if g is None:
        from ..config import get_config
        g = config.get("geomech.default.g", 9.81) if config else get_config("geomech.default.g", 9.81)
    
    depth = np.asarray(depth, dtype=np.float64)
    rhob = np.asarray(rhob, dtype=np.float64)
    
    if len(depth) == 0 or len(rhob) == 0:
        raise ValueError("Depth and density arrays must not be empty")
    
    if len(depth) != len(rhob):
        raise ValueError("Depth and density arrays must have same length")
    
    # Convert g/cc to kg/m³
    rhob_kg = rhob * 1000.0
    
    # Call optimized kernel
    return _calculate_overburden_stress_kernel(depth, rhob_kg, g)


def calculate_hydrostatic_pressure(
    depth: Union[np.ndarray, pd.Series],
    rho_water: Optional[float] = None,
    g: Optional[float] = None,
    config: Optional['ConfigManager'] = None
) -> np.ndarray:
    """
    Calculate hydrostatic pressure.
    
    Ph = rho_water * g * depth
    
    Args:
        depth: Depth array (meters)
        rho_water: Water density (g/cc). If None, reads from config.
        g: Gravitational acceleration (m/s²). If None, reads from config.
        config: ConfigManager instance. If None, uses global config.
        
    Returns:
        Hydrostatic pressure (MPa) as numpy array
    """
    # Load from config if not provided
    if rho_water is None:
        from ..config import get_config
        rho_water = config.get("geomech.default.rho_water", 1.03) if config else get_config("geomech.default.rho_water", 1.03)
    if g is None:
        from ..config import get_config
        g = config.get("geomech.default.g", 9.81) if config else get_config("geomech.default.g", 9.81)
    
    # Load from config if not provided
    if rho_water is None:
        from ..config import get_config
        rho_water = config.get("geomech.default.rho_water", 1.03) if config else get_config("geomech.default.rho_water", 1.03)
    if g is None:
        from ..config import get_config
        g = config.get("geomech.default.g", 9.81) if config else get_config("geomech.default.g", 9.81)
    
    depth = np.asarray(depth, dtype=np.float64)
    
    if len(depth) == 0:
        raise ValueError("Depth array must not be empty")
    
    rho_water_kg = rho_water * 1000  # Convert to kg/m³
    ph = rho_water_kg * g * depth / 1e6  # Convert Pa to MPa
    
    return ph


def calculate_pore_pressure_eaton(
    depth: Union[np.ndarray, pd.Series],
    dt: Union[np.ndarray, pd.Series],
    dt_normal: Union[np.ndarray, pd.Series],
    sv: Union[np.ndarray, pd.Series],
    ph: Union[np.ndarray, pd.Series],
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
        exponent: Eaton exponent, default 3.0 (typically 3.0 for sonic)
        
    Returns:
        Pore pressure (MPa) as numpy array
    """
    depth = np.asarray(depth, dtype=float)
    dt = np.asarray(dt, dtype=float)
    dt_normal = np.asarray(dt_normal, dtype=float)
    sv = np.asarray(sv, dtype=float)
    ph = np.asarray(ph, dtype=float)
    
    arrays = [depth, dt, dt_normal, sv, ph]
    if any(len(arr) == 0 for arr in arrays):
        raise ValueError("All input arrays must not be empty")
    
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Got lengths: {lengths}"
        )
    
    # Avoid division by zero
    dt = np.where(dt <= 0, np.nan, dt)
    dt_normal = np.where(dt_normal <= 0, np.nan, dt_normal)
    
    pp = sv - (sv - ph) * (dt_normal / dt) ** exponent
    
    return pp


def calculate_pore_pressure_bowers(
    depth: Union[np.ndarray, pd.Series],
    dt: Union[np.ndarray, pd.Series],
    dt_ml: float = 100.0,
    A: float = 5.0,
    B: float = 1.2,
    sv: Optional[Union[np.ndarray, pd.Series]] = None,
    ph: Optional[Union[np.ndarray, pd.Series]] = None,
    rho_water: float = 1.03,
    g: float = 9.81
) -> np.ndarray:
    """
    Calculate pore pressure using Bowers' method.
    
    This method accounts for unloading due to uplift/erosion.
    
    Args:
        depth: Depth array (meters)
        dt: Measured sonic transit time (us/ft)
        dt_ml: Mudline sonic (us/ft), default 100.0
        A: Bowers A parameter, default 5.0
        B: Bowers B parameter, default 1.2
        sv: Overburden stress (MPa), calculated if not provided
        ph: Hydrostatic pressure (MPa), calculated if not provided
        rho_water: Water density (g/cc), default 1.03
        g: Gravitational acceleration (m/s²), default 9.81
        
    Returns:
        Pore pressure (MPa) as numpy array
    """
    depth = np.asarray(depth, dtype=float)
    dt = np.asarray(dt, dtype=float)
    
    if len(depth) == 0 or len(dt) == 0:
        raise ValueError("Depth and sonic transit time arrays must not be empty")
    
    if len(depth) != len(dt):
        raise ValueError("Depth and sonic transit time arrays must have same length")
    
    if sv is None:
        # Assume average overburden gradient
        sv = 0.023 * depth  # MPa
    else:
        sv = np.asarray(sv, dtype=float)
        if len(sv) != len(depth):
            raise ValueError("Overburden stress array must have same length as depth")
    
    if ph is None:
        ph = calculate_hydrostatic_pressure(depth, rho_water, g)
    else:
        ph = np.asarray(ph, dtype=float)
        if len(ph) != len(depth):
            raise ValueError("Hydrostatic pressure array must have same length as depth")
    
    # Calculate effective stress from sonic
    sigma_eff = ((dt - dt_ml) / A) ** (1 / B)
    
    # Pore pressure
    pp = sv - sigma_eff
    
    return pp


def create_pressure_dataframe(
    depth: Union[np.ndarray, pd.Series],
    rhob: Optional[Union[np.ndarray, pd.Series]] = None,
    sv: Optional[Union[np.ndarray, pd.Series]] = None,
    ph: Optional[Union[np.ndarray, pd.Series]] = None,
    pp: Optional[Union[np.ndarray, pd.Series]] = None,
    rho_water: float = 1.03,
    g: float = 9.81
) -> pd.DataFrame:
    """
    Create a DataFrame with all pressure calculations.
    
    Args:
        depth: Depth array (meters)
        rhob: Bulk density array (g/cc), optional, calculated if not provided
        sv: Overburden stress (MPa), optional, calculated if not provided
        ph: Hydrostatic pressure (MPa), optional, calculated if not provided
        pp: Pore pressure (MPa), optional, uses hydrostatic if not provided
        rho_water: Water density (g/cc), default 1.03
        g: Gravitational acceleration (m/s²), default 9.81
        
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

