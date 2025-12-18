"""
Utility modules for GeoSuite.
"""

from .numba_helpers import njit, prange, NUMBA_AVAILABLE

__all__ = ['njit', 'prange', 'NUMBA_AVAILABLE']

