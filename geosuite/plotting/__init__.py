"""
Plotting utilities for GeoSuite.

This module provides functions for creating various plots and visualizations
including strip charts, crossplots, 3D viewers, and geospatial maps.
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
    'add_facies_track',
]

# Import ternary plotting functions
try:
    from .ternary import (
        ternary_plot,
        sand_silt_clay_plot,
        qfl_plot,
        mineral_composition_plot
    )
    __all__.extend([
        'ternary_plot',
        'sand_silt_clay_plot',
        'qfl_plot',
        'mineral_composition_plot'
    ])
except ImportError:
    pass

# Make interactive 3D visualization optional
try:
    from .interactive_3d import (
        create_3d_well_log_viewer,
        create_multi_well_3d_viewer,
        create_cross_section_viewer,
    )
    __all__.extend([
        'create_3d_well_log_viewer',
        'create_multi_well_3d_viewer',
        'create_cross_section_viewer',
    ])
except ImportError:
    pass

# Make geospatial mapping optional
try:
    from .geospatial import (
        create_field_map,
        create_well_trajectory_map,
    )
    __all__.extend([
        'create_field_map',
        'create_well_trajectory_map',
    ])
except ImportError:
    pass

# Make interactive Folium maps optional
try:
    from .interactive import (
        create_interactive_kriging_map,
        create_interactive_well_map,
        create_combined_map
    )
    __all__.extend([
        'create_interactive_kriging_map',
        'create_interactive_well_map',
        'create_combined_map'
    ])
except ImportError:
    pass

