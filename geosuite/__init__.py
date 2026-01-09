"""
GeoSuite - A Comprehensive Python Library for Geoscience Workflows
===================================================================

GeoSuite provides tools for petroleum engineering and geoscience applications:

**Core Modules:**
    - ``io``: Data import/export (LAS, SEG-Y, PPDM, WITSML, CSV)
    - ``petro``: Petrophysics calculations and crossplots
    - ``geomech``: Geomechanics stress and pressure calculations
    - ``ml``: Machine learning for facies classification
    - ``stratigraphy``: Automated stratigraphic interpretation (change-point detection)
    - ``imaging``: Core image processing and analysis
    - ``geospatial``: Geospatial operations with Apache Sedona
    - ``plotting``: Visualization utilities (strip charts, crossplots, ternary plots)
    - ``data``: Demo datasets and data loaders
    - ``modeling``: Spatial reservoir modeling with pygeomodeling integration (optional)
    - ``mining``: Ore geomodeling and block model generation (IDW + ML)
    - ``forecasting``: Production forecasting and decline curve analysis
    - ``base``: Base classes for consistent API patterns
    - ``utils``: Utility functions (Numba helpers, uncertainty quantification)

**Quick Start:**

    Import the library::

        import geosuite

    Or import specific modules::

        from geosuite import io, petro, geomech, ml
        from geosuite.data import load_demo_well_logs

**Examples:**

    Load demo well log data::

        from geosuite.data import load_demo_well_logs
        df = load_demo_well_logs()

    Calculate water saturation::

        from geosuite.petro import calculate_water_saturation
        sw = calculate_water_saturation(
            resistivity=10.5,
            porosity=0.25,
            formation_factor=1.0
        )

    Create a Pickett plot::

        from geosuite.petro import pickett_plot
        fig = pickett_plot(df, resistivity_col='RESDEEP', porosity_col='PHIE')

    Calculate overburden stress::

        from geosuite.geomech import calculate_overburden_stress
        sv = calculate_overburden_stress(depths, densities)

    Train a facies classifier::

        from geosuite.ml import train_facies_classifier
        from geosuite.data import load_facies_training_data

        df = load_facies_training_data()
        results = train_facies_classifier(
            df,
            feature_cols=['GR', 'NPHI', 'RHOB', 'PE'],
            target_col='Facies',
            model_type='random_forest'
        )

For more information, see the documentation at:
https://github.com/kylejones200/geosuite
"""

__version__ = "0.1.4"
__author__ = "K. Jones"
__license__ = "MIT"

# Import submodules for easier access
from geosuite import io
from geosuite import petro
from geosuite import geomech
from geosuite import ml
from geosuite import stratigraphy
from geosuite import data
from geosuite import plotting
from geosuite import config

try:
    from geosuite import analysis
    _has_analysis = True
except ImportError:
    _has_analysis = False
    analysis = None

try:
    from geosuite import forecasting
    _has_forecasting = True
except ImportError:
    _has_forecasting = False
    forecasting = None

# Conditional imports for optional dependencies
try:
    from geosuite import imaging
    _has_imaging = True
except ImportError:
    _has_imaging = False
    imaging = None

try:
    from geosuite import geospatial
    _has_geospatial = True
except ImportError:
    _has_geospatial = False
    geospatial = None

try:
    from geosuite import modeling
    _has_modeling = True
except ImportError:
    _has_modeling = False
    modeling = None

try:
    from geosuite import mining
    _has_mining = True
except ImportError:
    _has_mining = False
    mining = None

# Export commonly used functions at top level for convenience
from geosuite.data import (
    load_demo_well_logs,
    load_demo_facies,
    load_facies_training_data,
    load_kansas_training_wells,
)

from geosuite.petro import (
    calculate_water_saturation,
    pickett_plot,
    buckles_plot,
)

from geosuite.geomech import (
    calculate_overburden_stress,
    calculate_hydrostatic_pressure,
    calculate_effective_stress,
)

from geosuite.ml import (
    train_facies_classifier,
    MLflowFaciesClassifier,
)

from geosuite.plotting import (
    create_strip_chart,
    create_facies_log_plot,
)

from geosuite.stratigraphy import (
    preprocess_log,
    detect_pelt,
    detect_bayesian_online,
    compare_methods,
    find_consensus,
)

from geosuite.workflows import (
    run_workflow,
    load_workflow,
    WorkflowOrchestrator,
)

from geosuite.config import (
    ConfigManager,
    load_config,
    get_config,
    set_config,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # Submodules
    "io",
    "petro",
    "geomech",
    "ml",
    "stratigraphy",
    "imaging",
    "data",
    "plotting",
    "geospatial",
    "modeling",
    "mining",
    "analysis",
    "forecasting",
    "workflows",
    "config",
    
    # Data loaders (most commonly used)
    "load_demo_well_logs",
    "load_demo_facies",
    "load_facies_training_data",
    "load_kansas_training_wells",
    
    # Petrophysics (most commonly used)
    "calculate_water_saturation",
    "pickett_plot",
    "buckles_plot",
    
    # Geomechanics (most commonly used)
    "calculate_overburden_stress",
    "calculate_hydrostatic_pressure",
    "calculate_effective_stress",
    
    # Workflows
    "run_workflow",
    "load_workflow",
    "WorkflowOrchestrator",
    
    # Config
    "ConfigManager",
    "load_config",
    "get_config",
    "set_config",
    
    # Machine Learning (most commonly used)
    "train_facies_classifier",
    "MLflowFaciesClassifier",
    
    # Plotting (most commonly used)
    "create_strip_chart",
    "create_facies_log_plot",
    
    # Stratigraphy (most commonly used)
    "preprocess_log",
    "detect_pelt",
    "detect_bayesian_online",
    "compare_methods",
    "find_consensus",
]


def get_version():
    """Return the current version of GeoSuite."""
    return __version__


def get_info():
    """Return information about GeoSuite installation."""
    info = {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "modules": {
            "io": True,
            "petro": True,
            "geomech": True,
            "ml": True,
            "stratigraphy": True,
            "imaging": _has_imaging,
            "data": True,
            "plotting": True,
            "geospatial": _has_geospatial,
            "modeling": _has_modeling,
        }
    }
    return info


def list_demo_datasets():
    """List all available demo datasets."""
    import logging
    from geosuite import data
    
    logger = logging.getLogger(__name__)
    
    datasets = {
        "Well Logs": [
            "load_demo_well_logs() - Basic well log data",
            "load_field_data() - Multi-well field data",
        ],
        "Facies Classification": [
            "load_demo_facies() - Small facies demo",
            "load_facies_training_data() - Full training dataset (9 wells)",
            "load_facies_validation_data() - Validation dataset",
            "load_facies_vectors() - Complete facies vectors",
            "load_facies_well_data() - Full well log suite with facies",
            "load_kansas_training_wells() - Kansas University training set",
            "load_kansas_test_wells() - Kansas University test set",
        ],
    }
    
    logger.info("Available Demo Datasets in GeoSuite:")
    for category, items in datasets.items():
        logger.info(f"{category}:")
        for item in items:
            logger.info(f"  - {item}")
    
    return datasets
