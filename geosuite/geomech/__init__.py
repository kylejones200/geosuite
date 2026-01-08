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
from .stress_inversion import (
    invert_stress_from_breakout,
    invert_stress_from_dif,
    invert_stress_combined,
)
from .fracture_orientation import (
    predict_fracture_orientation,
    fracture_orientation_distribution,
    calculate_fracture_aperture,
    calculate_fracture_permeability,
)
from .failure_criteria import (
    mohr_coulomb_failure,
    drucker_prager_failure,
    hoek_brown_failure,
    griffith_failure,
    calculate_failure_envelope,
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
    
    # Stress inversion
    "invert_stress_from_breakout",
    "invert_stress_from_dif",
    "invert_stress_combined",
    
    # Fracture orientation
    "predict_fracture_orientation",
    "fracture_orientation_distribution",
    "calculate_fracture_aperture",
    "calculate_fracture_permeability",
    
    # Failure criteria
    "mohr_coulomb_failure",
    "drucker_prager_failure",
    "hoek_brown_failure",
    "griffith_failure",
    "calculate_failure_envelope",
]

