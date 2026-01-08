"""
Fracture orientation models for predicting natural and induced fracture patterns.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Tuple, Dict
from scipy.stats import vonmises

logger = logging.getLogger(__name__)


def predict_fracture_orientation(
    shmax_azimuth: Union[float, np.ndarray],
    shmin_azimuth: Union[float, np.ndarray],
    stress_ratio: Union[float, np.ndarray],
    method: str = 'coulomb'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Predict fracture orientation from stress field.
    
    Natural fractures typically form perpendicular to Shmin.
    Induced fractures form in direction of maximum stress.
    
    Args:
        shmax_azimuth: Maximum horizontal stress azimuth (degrees from north)
        shmin_azimuth: Minimum horizontal stress azimuth (degrees from north)
        stress_ratio: Ratio of (SHmax - Pp) / (Sv - Pp)
        method: Prediction method ('coulomb', 'griffith', 'tensile')
        
    Returns:
        Dictionary with fracture strike, dip, and type
    """
    shmax_azimuth = np.asarray(shmax_azimuth, dtype=float)
    shmin_azimuth = np.asarray(shmin_azimuth, dtype=float)
    stress_ratio = np.asarray(stress_ratio, dtype=float)
    
    model_configs = {
        'coulomb': _predict_coulomb_fracture,
        'griffith': _predict_griffith_fracture,
        'tensile': _predict_tensile_fracture,
    }
    
    if method not in model_configs:
        raise ValueError(f"Unknown method: {method}. Choose: {', '.join(model_configs.keys())}")
    
    return model_configs[method](shmax_azimuth, shmin_azimuth, stress_ratio)


def _predict_coulomb_fracture(
    shmax_azimuth: np.ndarray,
    shmin_azimuth: np.ndarray,
    stress_ratio: np.ndarray
) -> Dict[str, np.ndarray]:
    """Predict fracture orientation using Coulomb failure criterion."""
    # Natural fractures form at angle to Shmax based on friction
    friction_angle = 30.0  # degrees
    fracture_angle = 45.0 - friction_angle / 2.0
    
    # Fracture strike is perpendicular to Shmin for normal faulting
    # For strike-slip, fractures form at angle to Shmax
    is_normal = stress_ratio < 1.0
    is_strike_slip = (stress_ratio >= 1.0) & (stress_ratio < 1.5)
    
    fracture_strike = np.where(
        is_normal,
        shmin_azimuth + 90.0,  # Perpendicular to Shmin
        np.where(
            is_strike_slip,
            shmax_azimuth + fracture_angle,  # At angle to Shmax
            shmax_azimuth  # Parallel to Shmax for reverse
        )
    ) % 360.0
    
    # Fracture dip depends on stress regime
    fracture_dip = np.where(
        is_normal,
        60.0,  # Steep dip for normal faulting
        np.where(
            is_strike_slip,
            90.0,  # Vertical for strike-slip
            30.0  # Shallow for reverse
        )
    )
    
    fracture_type = np.where(
        is_normal,
        'normal',
        np.where(is_strike_slip, 'strike_slip', 'reverse')
    )
    
    return {
        'strike': fracture_strike,
        'dip': fracture_dip,
        'type': fracture_type,
        'azimuth': fracture_strike
    }


def _predict_griffith_fracture(
    shmax_azimuth: np.ndarray,
    shmin_azimuth: np.ndarray,
    stress_ratio: np.ndarray
) -> Dict[str, np.ndarray]:
    """Predict fracture orientation using Griffith failure criterion."""
    # Griffith theory: fractures form when tensile stress exceeds strength
    # Typically forms perpendicular to minimum principal stress
    fracture_strike = (shmin_azimuth + 90.0) % 360.0
    fracture_dip = np.full_like(fracture_strike, 90.0)  # Vertical fractures
    fracture_type = np.full(len(fracture_strike), 'tensile', dtype=object)
    
    return {
        'strike': fracture_strike,
        'dip': fracture_dip,
        'type': fracture_type,
        'azimuth': fracture_strike
    }


def _predict_tensile_fracture(
    shmax_azimuth: np.ndarray,
    shmin_azimuth: np.ndarray,
    stress_ratio: np.ndarray
) -> Dict[str, np.ndarray]:
    """Predict fracture orientation for tensile failure."""
    # Tensile fractures form perpendicular to minimum stress
    fracture_strike = (shmin_azimuth + 90.0) % 360.0
    fracture_dip = np.full_like(fracture_strike, 90.0)
    fracture_type = np.full(len(fracture_strike), 'tensile', dtype=object)
    
    return {
        'strike': fracture_strike,
        'dip': fracture_dip,
        'type': fracture_type,
        'azimuth': fracture_strike
    }


def fracture_orientation_distribution(
    mean_strike: float,
    concentration: float = 10.0,
    n_samples: int = 1000
) -> np.ndarray:
    """
    Generate fracture orientation distribution using von Mises distribution.
    
    Useful for modeling natural fracture networks with preferred orientation.
    
    Args:
        mean_strike: Mean fracture strike in degrees
        concentration: Concentration parameter (higher = more clustered)
        n_samples: Number of samples to generate
        
    Returns:
        Array of fracture strikes in degrees
    """
    kappa = concentration
    strikes_rad = vonmises.rvs(kappa, loc=np.radians(mean_strike), size=n_samples)
    strikes_deg = np.degrees(strikes_rad) % 360.0
    
    return strikes_deg


def calculate_fracture_aperture(
    normal_stress: Union[float, np.ndarray],
    closure_stress: float = 5.0,
    initial_aperture: float = 0.1,
    stiffness: float = 10.0
) -> Union[float, np.ndarray]:
    """
    Calculate fracture aperture under normal stress.
    
    Uses linear elastic model: aperture decreases with increasing normal stress.
    
    Args:
        normal_stress: Normal stress acting on fracture (MPa)
        closure_stress: Stress at which fracture closes (MPa)
        initial_aperture: Initial aperture at zero stress (mm)
        stiffness: Fracture stiffness (MPa/mm)
        
    Returns:
        Fracture aperture in mm
    """
    normal_stress = np.asarray(normal_stress, dtype=float)
    
    # Linear closure model
    aperture = initial_aperture - (normal_stress - closure_stress) / stiffness
    aperture = np.clip(aperture, 0.0, initial_aperture)
    
    return aperture


def calculate_fracture_permeability(
    aperture: Union[float, np.ndarray],
    spacing: float = 1.0,
    viscosity: float = 1.0e-3
) -> Union[float, np.ndarray]:
    """
    Calculate fracture permeability using cubic law.
    
    k = (aperture^3) / (12 * spacing)
    
    Args:
        aperture: Fracture aperture (mm)
        spacing: Fracture spacing (m)
        viscosity: Fluid viscosity (Pa·s)
        
    Returns:
        Permeability in mD
    """
    aperture = np.asarray(aperture, dtype=float)
    aperture_m = aperture * 1e-3  # Convert mm to m
    
    # Cubic law: k = b^3 / (12 * s)
    k_m2 = (aperture_m ** 3) / (12 * spacing)
    
    # Convert to mD
    k_md = k_m2 * 1.01325e15
    
    return k_md

