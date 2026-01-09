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
from .shaly_sand import (
    calculate_water_saturation_simandoux,
    calculate_water_saturation_indonesia,
    calculate_water_saturation_waxman_smits,
)
from .rock_physics import (
    gassmann_fluid_substitution,
    calculate_fluid_bulk_modulus,
    calculate_density_from_velocity,
)
from .permeability import (
    calculate_permeability_kozeny_carman,
    calculate_permeability_timur,
    calculate_permeability_wyllie_rose,
    calculate_permeability_coates_dumanoir,
    calculate_permeability_tixier,
    calculate_permeability_porosity_only,
)
from .avo import (
    calculate_velocities_from_slowness,
    preprocess_avo_inputs,
    calculate_avo_attributes,
    calculate_avo_from_slowness,
)
from .seismic_processing import (
    compute_hilbert_attributes,
    estimate_residual_phase,
    apply_phase_shift,
    correct_trace_phase,
    load_trace_from_segy,
)

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
    
    # Shaly sand models
    "calculate_water_saturation_simandoux",
    "calculate_water_saturation_indonesia",
    "calculate_water_saturation_waxman_smits",
    
    # Rock physics
    "gassmann_fluid_substitution",
    "calculate_fluid_bulk_modulus",
    "calculate_density_from_velocity",
    
    # Permeability estimation
    "calculate_permeability_kozeny_carman",
    "calculate_permeability_timur",
    "calculate_permeability_wyllie_rose",
    "calculate_permeability_coates_dumanoir",
    "calculate_permeability_tixier",
    "calculate_permeability_porosity_only",
    
    # AVO attributes
    "calculate_velocities_from_slowness",
    "preprocess_avo_inputs",
    "calculate_avo_attributes",
    "calculate_avo_from_slowness",
    
    # Seismic processing
    "compute_hilbert_attributes",
    "estimate_residual_phase",
    "apply_phase_shift",
    "correct_trace_phase",
    "load_trace_from_segy",
]
