"""
Mining and ore geomodeling module for GeoSuite.

This module provides tools for:
- Drillhole data processing and integration
- Inverse Distance Weighted (IDW) interpolation
- Hybrid IDW + Machine Learning block model generation
- Block model export for mine planning software
- Geostatistical modeling (variogram, kriging, SGS)
- Uncertainty quantification through Sequential Gaussian Simulation
- Exceedance probability calculations
- Ore grade forecasting (Ordinary Kriging, Gaussian Process, XGBoost)
- Feature engineering for spatial modeling
"""

__all__ = []

try:
    from .drillhole import (
        find_column,
        process_drillhole_data,
        merge_collar_assay,
        compute_3d_coordinates
    )
    __all__.extend([
        'find_column',
        'process_drillhole_data',
        'merge_collar_assay',
        'compute_3d_coordinates'
    ])
except ImportError:
    pass

try:
    from .interpolation import (
        idw_interpolate,
        compute_idw_residuals
    )
    __all__.extend([
        'idw_interpolate',
        'compute_idw_residuals'
    ])
except ImportError:
    pass

try:
    from .features import (
        build_spatial_features,
        build_block_model_features
    )
    __all__.extend([
        'build_spatial_features',
        'build_block_model_features'
    ])
except ImportError:
    pass

try:
    from .block_model import (
        create_block_model_grid,
        export_block_model
    )
    __all__.extend([
        'create_block_model_grid',
        'export_block_model'
    ])
except ImportError:
    pass

try:
    from .ore_modeling import (
        HybridOreModel,
        train_hybrid_model,
        predict_block_grades
    )
    __all__.extend([
        'HybridOreModel',
        'train_hybrid_model',
        'predict_block_grades'
    ])
except ImportError:
    pass

try:
    from .geostatistics import (
        project_to_utm,
        create_geochemical_dataframe,
        compute_variogram,
        ordinary_kriging,
        sequential_gaussian_simulation,
        compute_simulation_statistics,
        compute_exceedance_probability,
        log_transform,
        exp_transform
    )
    __all__.extend([
        'project_to_utm',
        'create_geochemical_dataframe',
        'compute_variogram',
        'ordinary_kriging',
        'sequential_gaussian_simulation',
        'compute_simulation_statistics',
        'compute_exceedance_probability',
        'log_transform',
        'exp_transform'
    ])
except ImportError:
    pass

try:
    from .forecasting import (
        generate_synthetic_geochemical_data,
        prepare_spatial_features,
        create_spatial_folds,
        fit_variogram_for_forecasting,
        ordinary_kriging_predict,
        train_gaussian_process,
        train_xgboost,
        create_prediction_grid,
        analyze_uncertainty_calibration,
        compare_forecasting_methods
    )
    __all__.extend([
        'generate_synthetic_geochemical_data',
        'prepare_spatial_features',
        'create_spatial_folds',
        'fit_variogram_for_forecasting',
        'ordinary_kriging_predict',
        'train_gaussian_process',
        'train_xgboost',
        'create_prediction_grid',
        'analyze_uncertainty_calibration',
        'compare_forecasting_methods'
    ])
except ImportError:
    pass

