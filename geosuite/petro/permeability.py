"""
Empirical permeability estimation models.

Common relationships for estimating permeability from porosity, water saturation,
and other petrophysical properties.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def calculate_permeability_kozeny_carman(
    phi: Union[np.ndarray, pd.Series],
    sw: Optional[Union[np.ndarray, pd.Series]] = None,
    grain_size_microns: float = 100.0,
    shape_factor: float = 2.5,
    tortuosity: float = 1.0
) -> np.ndarray:
    """
    Calculate permeability using Kozeny-Carman equation.
    
    k = (phi^3 * d^2) / (180 * (1 - phi)^2 * tau^2 * F)
    
    Where:
    - k: permeability (mD)
    - phi: porosity (fraction)
    - d: grain size (microns)
    - tau: tortuosity
    - F: shape factor
    
    Args:
        phi: Porosity (fraction, 0-1)
        sw: Water saturation (fraction, 0-1), optional
        grain_size_microns: Average grain size in microns (default 100)
        shape_factor: Shape factor (default 2.5)
        tortuosity: Tortuosity factor (default 1.0)
        
    Returns:
        Permeability in millidarcies (mD)
    """
    phi = np.asarray(phi, dtype=float)
    
    if len(phi) == 0:
        raise ValueError("Porosity array must not be empty")
    
    if np.any((phi < 0) | (phi > 1)):
        logger.warning("Porosity values outside [0, 1] range detected")
    
    phi = np.clip(phi, 0.01, 0.99)
    
    d_meters = grain_size_microns * 1e-6
    d_cm = d_meters * 100
    
    k_cm2 = (phi ** 3 * d_cm ** 2) / (180 * (1 - phi) ** 2 * tortuosity ** 2 * shape_factor)
    
    k_md = k_cm2 * 1.01325e8
    
    if sw is not None:
        sw = np.asarray(sw, dtype=float)
        if len(sw) != len(phi):
            raise ValueError("Water saturation and porosity arrays must have same length")
        k_md = k_md * (1 - sw) ** 2
    
    k_md = np.clip(k_md, 0.001, 1e6)
    
    return k_md


def calculate_permeability_timur(
    phi: Union[np.ndarray, pd.Series],
    sw: Union[np.ndarray, pd.Series],
    coefficient: float = 0.136,
    porosity_exponent: float = 4.4,
    saturation_exponent: float = 2.0
) -> np.ndarray:
    """
    Calculate permeability using Timur equation.
    
    k = C * (phi^a) / (sw^b)
    
    Where:
    - k: permeability (mD)
    - phi: porosity (fraction)
    - sw: water saturation (fraction)
    - C, a, b: empirical coefficients
    
    Default coefficients from Timur (1968) for sandstones.
    
    Args:
        phi: Porosity (fraction, 0-1)
        sw: Water saturation (fraction, 0-1)
        coefficient: Coefficient C (default 0.136)
        porosity_exponent: Exponent a (default 4.4)
        saturation_exponent: Exponent b (default 2.0)
        
    Returns:
        Permeability in millidarcies (mD)
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    
    if len(phi) == 0 or len(sw) == 0:
        raise ValueError("Porosity and water saturation arrays must not be empty")
    
    if len(phi) != len(sw):
        raise ValueError("Porosity and water saturation arrays must have same length")
    
    validation_checks = {
        'porosity outside [0, 1]': np.any((phi < 0) | (phi > 1)),
        'saturation outside [0, 1]': np.any((sw < 0) | (sw > 1)),
        'non-positive porosity': np.any(phi <= 0),
        'non-positive saturation': np.any(sw <= 0),
    }
    
    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, may result in invalid values")
    
    phi = np.clip(phi, 0.01, 0.99)
    sw = np.clip(sw, 0.01, 0.99)
    
    k_md = coefficient * (phi ** porosity_exponent) / (sw ** saturation_exponent)
    
    k_md = np.clip(k_md, 0.001, 1e6)
    
    return k_md


def calculate_permeability_wyllie_rose(
    phi: Union[np.ndarray, pd.Series],
    sw: Union[np.ndarray, pd.Series],
    coefficient: float = 0.625,
    porosity_exponent: float = 6.0,
    saturation_exponent: float = 2.0
) -> np.ndarray:
    """
    Calculate permeability using Wyllie-Rose equation.
    
    k = C * (phi^a) / (sw^b)
    
    Similar to Timur but with different coefficients.
    Default coefficients from Wyllie & Rose (1950).
    
    Args:
        phi: Porosity (fraction, 0-1)
        sw: Water saturation (fraction, 0-1)
        coefficient: Coefficient C (default 0.625)
        porosity_exponent: Exponent a (default 6.0)
        saturation_exponent: Exponent b (default 2.0)
        
    Returns:
        Permeability in millidarcies (mD)
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    
    if len(phi) == 0 or len(sw) == 0:
        raise ValueError("Porosity and water saturation arrays must not be empty")
    
    if len(phi) != len(sw):
        raise ValueError("Porosity and water saturation arrays must have same length")
    
    phi = np.clip(phi, 0.01, 0.99)
    sw = np.clip(sw, 0.01, 0.99)
    
    k_md = coefficient * (phi ** porosity_exponent) / (sw ** saturation_exponent)
    
    k_md = np.clip(k_md, 0.001, 1e6)
    
    return k_md


def calculate_permeability_coates_dumanoir(
    phi: Union[np.ndarray, pd.Series],
    sw: Union[np.ndarray, pd.Series],
    coefficient: float = 70.0,
    porosity_exponent: float = 2.0,
    saturation_exponent: float = 2.0
) -> np.ndarray:
    """
    Calculate permeability using Coates-Dumanoir equation.
    
    k = C * (phi^a) * ((1 - sw) / sw)^b
    
    Where irreducible water saturation is accounted for.
    Default coefficients from Coates & Dumanoir (1973).
    
    Args:
        phi: Porosity (fraction, 0-1)
        sw: Water saturation (fraction, 0-1)
        coefficient: Coefficient C (default 70.0)
        porosity_exponent: Exponent a (default 2.0)
        saturation_exponent: Exponent b (default 2.0)
        
    Returns:
        Permeability in millidarcies (mD)
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    
    if len(phi) == 0 or len(sw) == 0:
        raise ValueError("Porosity and water saturation arrays must not be empty")
    
    if len(phi) != len(sw):
        raise ValueError("Porosity and water saturation arrays must have same length")
    
    phi = np.clip(phi, 0.01, 0.99)
    sw = np.clip(sw, 0.01, 0.99)
    
    k_md = coefficient * (phi ** porosity_exponent) * ((1 - sw) / sw) ** saturation_exponent
    
    k_md = np.clip(k_md, 0.001, 1e6)
    
    return k_md


def calculate_permeability_tixier(
    phi: Union[np.ndarray, pd.Series],
    sw: Union[np.ndarray, pd.Series],
    coefficient: float = 0.136,
    porosity_exponent: float = 4.4,
    saturation_exponent: float = 2.0
) -> np.ndarray:
    """
    Calculate permeability using Tixier equation.
    
    k = C * (phi^a) / (sw^b)
    
    Similar to Timur equation. Default coefficients from Tixier (1949).
    
    Args:
        phi: Porosity (fraction, 0-1)
        sw: Water saturation (fraction, 0-1)
        coefficient: Coefficient C (default 0.136)
        porosity_exponent: Exponent a (default 4.4)
        saturation_exponent: Exponent b (default 2.0)
        
    Returns:
        Permeability in millidarcies (mD)
    """
    return calculate_permeability_timur(phi, sw, coefficient, porosity_exponent, saturation_exponent)


def calculate_permeability_porosity_only(
    phi: Union[np.ndarray, pd.Series],
    coefficient: float = 100.0,
    exponent: float = 3.0
) -> np.ndarray:
    """
    Calculate permeability from porosity only (simple power law).
    
    k = C * phi^a
    
    Useful when water saturation is not available.
    Default coefficients are general approximations.
    
    Args:
        phi: Porosity (fraction, 0-1)
        coefficient: Coefficient C (default 100.0)
        exponent: Exponent a (default 3.0)
        
    Returns:
        Permeability in millidarcies (mD)
    """
    phi = np.asarray(phi, dtype=float)
    
    if len(phi) == 0:
        raise ValueError("Porosity array must not be empty")
    
    phi = np.clip(phi, 0.01, 0.99)
    
    k_md = coefficient * (phi ** exponent)
    
    k_md = np.clip(k_md, 0.001, 1e6)
    
    return k_md

