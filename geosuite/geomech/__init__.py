"""
Geomechanics module for GeoSuite.

Provides tools for:
- Pressure calculations (overburden, hydrostatic, pore pressure)
- Stress calculations (effective stress, stress ratios)
- Stress polygon analysis
- Pressure and stress profiling
"""

from .pressures import (
    calculate_overburden_stress,
    calculate_hydrostatic_pressure,
    calculate_pore_pressure_eaton,
    calculate_pore_pressure_bowers,
    create_pressure_dataframe
)
from .stresses import (
    calculate_effective_stress,
    calculate_overpressure,
    calculate_pressure_gradient,
    pressure_to_mud_weight,
    calculate_stress_ratio,
    estimate_shmin_from_poisson
)
from .stress_polygon import (
    stress_polygon_limits,
    plot_stress_polygon,
    determine_stress_regime
)
from .profiles import (
    plot_pressure_profile,
    plot_mud_weight_profile
)

__all__ = [
    # Pressure calculations
    "calculate_overburden_stress",
    "calculate_hydrostatic_pressure",
    "calculate_pore_pressure_eaton",
    "calculate_pore_pressure_bowers",
    "create_pressure_dataframe",
    
    # Stress calculations
    "calculate_effective_stress",
    "calculate_overpressure",
    "calculate_pressure_gradient",
    "pressure_to_mud_weight",
    "calculate_stress_ratio",
    "estimate_shmin_from_poisson",
    
    # Stress polygon
    "stress_polygon_limits",
    "plot_stress_polygon",
    "determine_stress_regime",
    
    # Profiles
    "plot_pressure_profile",
    "plot_mud_weight_profile",
]

