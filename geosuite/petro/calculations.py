"""
Petrophysical calculations (Archie, porosity, etc.).
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from ..config import get_config, ConfigManager


def calculate_water_saturation(
    phi: Union[np.ndarray, pd.Series],
    rt: Union[np.ndarray, pd.Series],
    rw: Optional[float] = None,
    m: Optional[float] = None,
    n: Optional[float] = None,
    a: Optional[float] = None,
    config: Optional[ConfigManager] = None
) -> np.ndarray:
    """
    Calculate water saturation using Archie's equation.
    
    Sw = ((a * Rw) / (phi^m * Rt))^(1/n)
    
    Args:
        phi: Porosity (fraction, 0-1)
        rt: True resistivity (ohm-m)
        rw: Formation water resistivity (ohm-m). If None, reads from config.
        m: Cementation exponent. If None, reads from config.
        n: Saturation exponent. If None, reads from config.
        a: Tortuosity factor. If None, reads from config.
        config: ConfigManager instance. If None, uses global config.
        
    Returns:
        Water saturation (fraction, 0-1) as numpy array
    """
    # Load from config if not provided
    if rw is None:
        rw = config.get("petro.archie.rw", 0.05) if config else get_config("petro.archie.rw", 0.05)
    if m is None:
        m = config.get("petro.archie.m", 2.0) if config else get_config("petro.archie.m", 2.0)
    if n is None:
        n = config.get("petro.archie.n", 2.0) if config else get_config("petro.archie.n", 2.0)
    if a is None:
        a = config.get("petro.archie.a", 1.0) if config else get_config("petro.archie.a", 1.0)
    
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    
    # Handle scalar inputs (0-d arrays) by converting to 1-d
    if phi.ndim == 0:
        phi = np.array([phi.item()])
    if rt.ndim == 0:
        rt = np.array([rt.item()])
    
    # Store original shape for reshaping output
    original_shape = phi.shape
    
    # Flatten to 1-d for length checks and calculations
    phi_flat = phi.flatten()
    rt_flat = rt.flatten()
    
    if len(phi_flat) == 0 or len(rt_flat) == 0:
        raise ValueError("Porosity and resistivity arrays must not be empty")
    
    if len(phi_flat) != len(rt_flat):
        raise ValueError("Porosity and resistivity arrays must have same length")
    
    phi_flat = np.where(phi_flat <= 0, np.nan, phi_flat)
    rt_flat = np.where(rt_flat <= 0, np.nan, rt_flat)
    
    sw_flat = ((a * rw) / (phi_flat ** m * rt_flat)) ** (1 / n)
    sw_flat = np.clip(sw_flat, 0, 1)
    
    # Reshape to match original input shape
    sw = sw_flat.reshape(original_shape)
    
    return sw


def calculate_porosity_from_density(
    rhob: Union[np.ndarray, pd.Series],
    rho_matrix: Optional[float] = None,
    rho_fluid: Optional[float] = None,
    config: Optional[ConfigManager] = None
) -> np.ndarray:
    """
    Calculate porosity from bulk density.
    
    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    
    Args:
        rhob: Bulk density (g/cc)
        rho_matrix: Matrix density (g/cc). If None, reads from config.
        rho_fluid: Fluid density (g/cc). If None, reads from config.
        config: ConfigManager instance. If None, uses global config.
        
    Returns:
        Porosity (fraction, 0-1) as numpy array
    """
    # Load from config if not provided
    if rho_matrix is None:
        rho_matrix = config.get("petro.density.rho_matrix", 2.65) if config else get_config("petro.density.rho_matrix", 2.65)
    if rho_fluid is None:
        rho_fluid = config.get("petro.density.rho_fluid", 1.0) if config else get_config("petro.density.rho_fluid", 1.0)
    
    rhob = np.asarray(rhob, dtype=float)
    
    if len(rhob) == 0:
        raise ValueError("Bulk density array must not be empty")
    
    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    phi = np.clip(phi, 0, 1)
    
    return phi


def calculate_formation_factor(
    phi: Union[np.ndarray, pd.Series],
    m: Optional[float] = None,
    a: Optional[float] = None,
    config: Optional[ConfigManager] = None
) -> np.ndarray:
    """
    Calculate formation resistivity factor.
    
    F = a / phi^m
    
    Args:
        phi: Porosity (fraction, 0-1)
        m: Cementation exponent. If None, reads from config.
        a: Tortuosity factor. If None, reads from config.
        config: ConfigManager instance. If None, uses global config.
        
    Returns:
        Formation factor as numpy array
    """
    # Load from config if not provided
    if m is None:
        m = config.get("petro.archie.m", 2.0) if config else get_config("petro.archie.m", 2.0)
    if a is None:
        a = config.get("petro.archie.a", 1.0) if config else get_config("petro.archie.a", 1.0)
    
    phi = np.asarray(phi, dtype=float)
    
    if len(phi) == 0:
        raise ValueError("Porosity array must not be empty")
    
    phi = np.where(phi <= 0, np.nan, phi)
    return a / (phi ** m)

