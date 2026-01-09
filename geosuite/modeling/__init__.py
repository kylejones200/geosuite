"""
Spatial reservoir modeling integration with pygeomodeling.

This module provides integration between GeoSuite well log analysis
and pygeomodeling's Gaussian Process and Kriging capabilities for
spatial reservoir property modeling.

Example:
    >>> from geosuite.modeling import WellLogToSpatial, SpatialPropertyModeler
    >>> from geosuite.io import load_las_file
    >>> 
    >>> # Load well log data
    >>> las = load_las_file('well_001.las')
    >>> df = las.df()
    >>> 
    >>> # Convert to spatial format
    >>> converter = WellLogToSpatial()
    >>> spatial_data = converter.convert(df, x=1000, y=2000, z='DEPTH')
    >>> 
    >>> # Model permeability spatially
    >>> modeler = SpatialPropertyModeler()
    >>> model = modeler.fit_permeability(spatial_data)
    >>> predictions = model.predict(grid_coordinates)
"""

import logging

logger = logging.getLogger(__name__)

# Attempt to import pygeomodeling components
try:
    from pygeomodeling import GRDECLParser, UnifiedSPE9Toolkit
    from pygeomodeling.model_gp import StandardGP, DeepGP
    
    PYGEO_AVAILABLE = True
    logger.info("✓ pygeomodeling integration available")
except ImportError:
    PYGEO_AVAILABLE = False
    logger.warning(
        "pygeomodeling not available. Install with: pip install pygeomodeling "
        "or pip install geosuite[modeling]"
    )
    GRDECLParser = None
    UnifiedSPE9Toolkit = None
    StandardGP = None
    DeepGP = None

# Import integration components
try:
    from .converters import WellLogToSpatial, SpatialDataConverter
    from .modelers import SpatialPropertyModeler, ReservoirModelBuilder
    from .workflows import create_reservoir_model, interpolate_properties
    
    __all__ = [
        "WellLogToSpatial",
        "SpatialDataConverter",
        "SpatialPropertyModeler",
        "ReservoirModelBuilder",
        "create_reservoir_model",
        "interpolate_properties",
        "PYGEO_AVAILABLE",
    ]
    
    if PYGEO_AVAILABLE:
        __all__.extend([
            "GRDECLParser",
            "UnifiedSPE9Toolkit",
            "StandardGP",
            "DeepGP",
        ])
        
except ImportError as e:
    logger.warning(f"Some modeling components not available: {e}")
    __all__ = ["PYGEO_AVAILABLE"]

# Import GPR modeling (independent of pygeomodeling)
try:
    from .gpr_modeling import (
        GPRReservoirModel,
        export_to_grdecl,
        predict_on_grid,
    )
    
    if "GPRReservoirModel" not in __all__:
        __all__.extend([
            "GPRReservoirModel",
            "export_to_grdecl",
            "predict_on_grid",
        ])
except ImportError as e:
    logger.warning(f"GPR modeling components not available: {e}")
