"""
Petrophysics module for GeoSuite.

Provides tools for:
- Archie equation calculations
- Petrophysical crossplots (Pickett, Buckles)
- Lithology identification
- Porosity and saturation calculations
"""

from .archie import ArchieParams, archie_sw, compute_bvw, pickett_isolines
from .calculations import (
    calculate_water_saturation,
    calculate_porosity_from_density,
    calculate_formation_factor
)
from .pickett import pickett_plot
from .buckles import buckles_plot
from .lithology import neutron_density_crossplot

__all__ = [
    # Archie module (existing)
    "ArchieParams",
    "archie_sw",
    "compute_bvw",
    "pickett_isolines",
    
    # Calculations
    "calculate_water_saturation",
    "calculate_porosity_from_density",
    "calculate_formation_factor",
    
    # Plotting
    "pickett_plot",
    "buckles_plot",
    "neutron_density_crossplot",
]
