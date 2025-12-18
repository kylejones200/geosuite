"""
Plotting utilities for GeoSuite.

This module provides functions for creating various plots and visualizations
including strip charts, crossplots, and other geoscience visualizations.
"""

from .strip_charts import (
    create_strip_chart,
    create_facies_log_plot,
    add_log_track,
    add_facies_track
)

__all__ = [
    'create_strip_chart',
    'create_facies_log_plot',
    'add_log_track',
    'add_facies_track'
]

