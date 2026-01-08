"""
Time series and network analysis for subsurface data.

Provides tools for converting well log time series to network representations
and analyzing their structural properties.
"""

__all__ = []

try:
    from .ts2net_integration import (
        WellLogNetworkAnalyzer,
        analyze_well_log_network,
        compare_wells_network,
        detect_formation_boundaries_network,
        extract_network_features,
    )
    __all__.extend([
        'WellLogNetworkAnalyzer',
        'analyze_well_log_network',
        'compare_wells_network',
        'detect_formation_boundaries_network',
        'extract_network_features',
    ])
except ImportError:
    pass

