"""
High-level workflows for integrating GeoSuite and pygeomodeling.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .converters import WellLogToSpatial, SpatialDataConverter
from .modelers import SpatialPropertyModeler, ReservoirModelBuilder

logger = logging.getLogger(__name__)


def create_reservoir_model(
    well_data: Dict[str, pd.DataFrame],
    coordinates: Dict[str, Tuple[float, float]],
    properties: List[str],
    z_col: str = 'DEPTH',
    model_type: str = 'gpr'
) -> ReservoirModelBuilder:
    """
    Create a complete reservoir model from multiple well logs.
    
    This is a convenience function that combines well log conversion,
    spatial modeling, and property prediction into a single workflow.
    
    Parameters
    ----------
    well_data : Dict[str, pd.DataFrame]
        Dictionary mapping well names to well log DataFrames
    coordinates : Dict[str, Tuple[float, float]]
        Dictionary mapping well names to (x, y) surface coordinates
    properties : List[str]
        List of property column names to model (e.g., ['PERMEABILITY', 'POROSITY'])
    z_col : str, default 'DEPTH'
        Column name for depth coordinate
    model_type : str, default 'gpr'
        Type of spatial model to use
        
    Returns
    -------
    ReservoirModelBuilder
        Configured reservoir model builder with all property models
        
    Example
    -------
    >>> from geosuite.modeling import create_reservoir_model
    >>> from geosuite.io import load_las_file
    >>> 
    >>> # Load multiple wells
    >>> wells = {
    ...     'Well_001': load_las_file('well_001.las').df(),
    ...     'Well_002': load_las_file('well_002.las').df(),
    ... }
    >>> 
    >>> # Define coordinates
    >>> coords = {
    ...     'Well_001': (1000.0, 2000.0),
    ...     'Well_002': (1500.0, 2500.0),
    ... }
    >>> 
    >>> # Create model
    >>> model = create_reservoir_model(
    ...     wells, coords, properties=['PERMEABILITY', 'POROSITY']
    ... )
    >>> 
    >>> # Predict on grid
    >>> grid = SpatialDataConverter.create_prediction_grid(
    ...     (0, 3000), (0, 4000), (0, 5000)
    ... )
    >>> predictions = model.predict_all_properties(grid)
    """
    converter = WellLogToSpatial()
    
    # Convert all wells to spatial format
    spatial_data = converter.convert_multiple_wells(
        well_data, coordinates, z=z_col, properties=properties
    )
    
    # Build reservoir model
    builder = ReservoirModelBuilder()
    
    for prop in properties:
        if prop in spatial_data.columns:
            builder.add_property_model(
                prop.lower(),
                spatial_data,
                prop,
                model_type=model_type
            )
        else:
            logger.warning(f"Property '{prop}' not found in spatial data, skipping")
    
    logger.info(
        f"Created reservoir model with {len(builder.models)} properties "
        f"from {len(well_data)} wells"
    )
    
    return builder


def interpolate_properties(
    spatial_data: pd.DataFrame,
    property_col: str,
    grid_coordinates: np.ndarray,
    model_type: str = 'gpr',
    return_uncertainty: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Interpolate a property onto a regular grid using spatial modeling.
    
    Parameters
    ----------
    spatial_data : pd.DataFrame
        Spatial DataFrame with coordinates and property values
    property_col : str
        Column name of property to interpolate
    grid_coordinates : np.ndarray
        Array of shape (n_points, 3) with (X, Y, Z) coordinates for grid
    model_type : str, default 'gpr'
        Type of spatial model to use
    return_uncertainty : bool, default True
        If True, also return uncertainty estimates
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        DataFrame with interpolated values at grid points,
        or (values, uncertainty) if return_uncertainty=True
        
    Example
    -------
    >>> from geosuite.modeling import interpolate_properties
    >>> from geosuite.modeling.converters import SpatialDataConverter
    >>> 
    >>> # Create grid
    >>> grid = SpatialDataConverter.create_prediction_grid(
    ...     (0, 3000), (0, 4000), (0, 5000), nx=50, ny=50, nz=20
    ... )
    >>> 
    >>> # Interpolate
    >>> predictions, uncertainty = interpolate_properties(
    ...     spatial_data, 'PERMEABILITY', grid, return_uncertainty=True
    ... )
    """
    # Fit model
    modeler = SpatialPropertyModeler(model_type=model_type)
    modeler.fit_property(spatial_data, property_col)
    
    # Predict
    if return_uncertainty:
        pred, std = modeler.predict(grid_coordinates, return_std=True)
        
        results_df = pd.DataFrame({
            'X': grid_coordinates[:, 0],
            'Y': grid_coordinates[:, 1],
            'Z': grid_coordinates[:, 2],
            property_col: pred.flatten() if pred.ndim > 1 else pred,
        })
        
        uncertainty_df = pd.DataFrame({
            'X': grid_coordinates[:, 0],
            'Y': grid_coordinates[:, 1],
            'Z': grid_coordinates[:, 2],
            f'{property_col}_STD': std.flatten() if std.ndim > 1 else std,
        })
        
        logger.info(
            f"Interpolated {property_col} to {len(grid_coordinates)} grid points "
            f"with uncertainty estimates"
        )
        
        return results_df, uncertainty_df
    else:
        pred = modeler.predict(grid_coordinates)
        
        results_df = pd.DataFrame({
            'X': grid_coordinates[:, 0],
            'Y': grid_coordinates[:, 1],
            'Z': grid_coordinates[:, 2],
            property_col: pred.flatten() if pred.ndim > 1 else pred,
        })
        
        logger.info(
            f"Interpolated {property_col} to {len(grid_coordinates)} grid points"
        )
        
        return results_df

