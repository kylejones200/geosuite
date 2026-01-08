"""
Advanced failure criteria for rock mechanics analysis.

Implements Mohr-Coulomb, Drucker-Prager, Hoek-Brown, and other failure criteria.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


def mohr_coulomb_failure(
    sigma1: Union[float, np.ndarray],
    sigma3: Union[float, np.ndarray],
    cohesion: float = 10.0,
    friction_angle: float = 30.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate Mohr-Coulomb failure criterion.
    
    sigma1_fail = sigma3 * tan^2(45 + phi/2) + 2 * c * tan(45 + phi/2)
    
    Args:
        sigma1: Maximum principal stress (MPa)
        sigma3: Minimum principal stress (MPa)
        cohesion: Cohesion (MPa)
        friction_angle: Friction angle in degrees
        
    Returns:
        Tuple of (failure_stress, safety_factor)
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)
    
    phi_rad = np.radians(friction_angle)
    tan_phi = np.tan(phi_rad)
    tan_squared = np.tan(np.radians(45.0 + friction_angle / 2.0)) ** 2
    
    # Failure stress
    sigma1_fail = sigma3 * tan_squared + 2 * cohesion * np.sqrt(tan_squared)
    
    # Safety factor
    safety_factor = sigma1_fail / sigma1
    
    return sigma1_fail, safety_factor


def drucker_prager_failure(
    sigma1: Union[float, np.ndarray],
    sigma2: Union[float, np.ndarray],
    sigma3: Union[float, np.ndarray],
    cohesion: float = 10.0,
    friction_angle: float = 30.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate Drucker-Prager failure criterion.
    
    Uses mean stress and deviatoric stress invariants.
    
    Args:
        sigma1: Maximum principal stress (MPa)
        sigma2: Intermediate principal stress (MPa)
        sigma3: Minimum principal stress (MPa)
        cohesion: Cohesion (MPa)
        friction_angle: Friction angle in degrees
        
    Returns:
        Tuple of (failure_stress, safety_factor)
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)
    
    phi_rad = np.radians(friction_angle)
    
    # Mean stress
    I1 = sigma1 + sigma2 + sigma3
    
    # Deviatoric stress
    J2 = ((sigma1 - sigma2) ** 2 + (sigma2 - sigma3) ** 2 + (sigma3 - sigma1) ** 2) / 6.0
    
    # Drucker-Prager parameters
    alpha = np.sin(phi_rad) / np.sqrt(3.0 * (3.0 + np.sin(phi_rad) ** 2))
    k = np.sqrt(3.0) * cohesion * np.cos(phi_rad) / np.sqrt(3.0 + np.sin(phi_rad) ** 2)
    
    # Failure criterion: sqrt(J2) = alpha * I1 + k
    sqrt_J2_fail = alpha * I1 + k
    sqrt_J2_actual = np.sqrt(J2)
    
    # Safety factor
    safety_factor = sqrt_J2_fail / sqrt_J2_actual
    
    return sqrt_J2_fail, safety_factor


def hoek_brown_failure(
    sigma1: Union[float, np.ndarray],
    sigma3: Union[float, np.ndarray],
    ucs: float = 50.0,
    mi: float = 15.0,
    gsi: float = 75.0,
    d: float = 0.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate Hoek-Brown failure criterion for rock masses.
    
    sigma1 = sigma3 + ucs * (mb * sigma3 / ucs + s)^a
    
    Args:
        sigma1: Maximum principal stress (MPa)
        sigma3: Minimum principal stress (MPa)
        ucs: Unconfined compressive strength (MPa)
        mi: Intact rock parameter
        gsi: Geological Strength Index (0-100)
        d: Disturbance factor (0-1)
        
    Returns:
        Tuple of (failure_stress, safety_factor)
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)
    
    # Hoek-Brown parameters
    mb = mi * np.exp((gsi - 100) / (28 - 14 * d))
    s = np.exp((gsi - 100) / (9 - 3 * d))
    a = 0.5 + (1.0 / 6.0) * (np.exp(-gsi / 15.0) - np.exp(-20.0 / 3.0))
    
    # Failure stress
    sigma1_fail = sigma3 + ucs * (mb * sigma3 / ucs + s) ** a
    
    # Safety factor
    safety_factor = sigma1_fail / sigma1
    
    return sigma1_fail, safety_factor


def griffith_failure(
    sigma1: Union[float, np.ndarray],
    sigma3: Union[float, np.ndarray],
    tensile_strength: float = 5.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate Griffith failure criterion for brittle materials.
    
    (sigma1 - sigma3)^2 = 8 * T0 * (sigma1 + sigma3) when sigma1 + 3*sigma3 > 0
    sigma3 = -T0 when sigma1 + 3*sigma3 < 0
    
    Args:
        sigma1: Maximum principal stress (MPa)
        sigma3: Minimum principal stress (MPa)
        tensile_strength: Tensile strength T0 (MPa)
        
    Returns:
        Tuple of (failure_stress, safety_factor)
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)
    
    # Check which branch of criterion applies
    condition = sigma1 + 3 * sigma3 > 0
    
    # Branch 1: (sigma1 - sigma3)^2 = 8 * T0 * (sigma1 + sigma3)
    sigma1_fail_1 = sigma3 + np.sqrt(8 * tensile_strength * (sigma1 + sigma3))
    
    # Branch 2: sigma3 = -T0
    sigma1_fail_2 = np.full_like(sigma1, -tensile_strength)
    
    sigma1_fail = np.where(condition, sigma1_fail_1, sigma1_fail_2)
    
    # Safety factor
    safety_factor = sigma1_fail / sigma1
    
    return sigma1_fail, safety_factor


def calculate_failure_envelope(
    sigma3_range: np.ndarray,
    criterion: str = 'mohr_coulomb',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate failure envelope for given stress range.
    
    Args:
        sigma3_range: Range of minimum principal stress values (MPa)
        criterion: Failure criterion ('mohr_coulomb', 'drucker_prager', 'hoek_brown', 'griffith')
        **kwargs: Parameters for failure criterion
        
    Returns:
        Tuple of (sigma3, sigma1_fail) arrays
    """
    sigma3_range = np.asarray(sigma3_range, dtype=float)
    
    criterion_configs = {
        'mohr_coulomb': lambda s3: mohr_coulomb_failure(
            np.zeros_like(s3), s3,
            kwargs.get('cohesion', 10.0),
            kwargs.get('friction_angle', 30.0)
        )[0],
        'hoek_brown': lambda s3: hoek_brown_failure(
            np.zeros_like(s3), s3,
            kwargs.get('ucs', 50.0),
            kwargs.get('mi', 15.0),
            kwargs.get('gsi', 75.0),
            kwargs.get('d', 0.0)
        )[0],
        'griffith': lambda s3: griffith_failure(
            np.zeros_like(s3), s3,
            kwargs.get('tensile_strength', 5.0)
        )[0],
    }
    
    if criterion not in criterion_configs:
        raise ValueError(f"Unknown criterion: {criterion}. Choose: {', '.join(criterion_configs.keys())}")
    
    sigma1_fail = criterion_configs[criterion](sigma3_range)
    
    return sigma3_range, sigma1_fail

