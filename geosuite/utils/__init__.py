"""
Utility modules for GeoSuite.
"""

from .numba_helpers import njit, prange, NUMBA_AVAILABLE
from .uncertainty import (
    propagate_error,
    confidence_interval,
    monte_carlo_uncertainty,
    uncertainty_porosity_from_density,
    uncertainty_water_saturation,
    uncertainty_permeability,
)

__all__ = [
    'njit', 
    'prange', 
    'NUMBA_AVAILABLE',
    'propagate_error',
    'confidence_interval',
    'monte_carlo_uncertainty',
    'uncertainty_porosity_from_density',
    'uncertainty_water_saturation',
    'uncertainty_permeability',
]

