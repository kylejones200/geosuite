"""
Stratigraphy module for automated formation interpretation.

This module provides tools for stratigraphic analysis including:
- Change-point detection for automated formation top picking
- Stratigraphic correlation
- Sequence stratigraphy analysis
"""

from .changepoint import (
    preprocess_log,
    detect_pelt,
    detect_bayesian_online,
    compare_methods,
    find_consensus,
    tune_penalty_to_target_count,
    RUPTURES_AVAILABLE
)

__all__ = [
    'preprocess_log',
    'detect_pelt',
    'detect_bayesian_online',
    'compare_methods',
    'find_consensus',
    'tune_penalty_to_target_count',
    'RUPTURES_AVAILABLE'
]


