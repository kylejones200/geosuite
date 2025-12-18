"""
Petrophysical calculations (Archie, porosity, etc.).
"""

import numpy as np


def calculate_water_saturation(
    phi: np.ndarray,
    rt: np.ndarray,
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0
) -> np.ndarray:
    """
    Calculate water saturation using Archie's equation.
    
    Sw = ((a * Rw) / (phi^m * Rt))^(1/n)
    
    Args:
        phi: Porosity (fraction)
        rt: True resistivity (ohm-m)
        rw: Formation water resistivity (ohm-m)
        m: Cementation exponent
        n: Saturation exponent
        a: Tortuosity factor
        
    Returns:
        Water saturation array
    """
    phi = np.where(phi <= 0, np.nan, phi)
    rt = np.where(rt <= 0, np.nan, rt)
    
    sw = ((a * rw) / (phi ** m * rt)) ** (1 / n)
    sw = np.clip(sw, 0, 1)
    
    return sw


def calculate_porosity_from_density(
    rhob: np.ndarray,
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0
) -> np.ndarray:
    """
    Calculate porosity from bulk density.
    
    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    
    Args:
        rhob: Bulk density (g/cc)
        rho_matrix: Matrix density (g/cc)
        rho_fluid: Fluid density (g/cc)
        
    Returns:
        Porosity array (fraction)
    """
    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    phi = np.clip(phi, 0, 1)
    
    return phi


def calculate_formation_factor(
    phi: np.ndarray,
    m: float = 2.0,
    a: float = 1.0
) -> np.ndarray:
    """
    Calculate formation resistivity factor.
    
    F = a / phi^m
    
    Args:
        phi: Porosity (fraction)
        m: Cementation exponent
        a: Tortuosity factor
        
    Returns:
        Formation factor array
    """
    phi = np.where(phi <= 0, np.nan, phi)
    return a / (phi ** m)

