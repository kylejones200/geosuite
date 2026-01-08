"""
Converters for transforming GeoSuite well log data to spatial formats
compatible with pygeomodeling.
"""
import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WellLogToSpatial:
    """
    Convert well log data from GeoSuite format to spatial coordinates
    for use with pygeomodeling.
    
    This class handles the transformation of 1D well log data (depth-based)
    into 3D spatial coordinates (x, y, z) required for spatial modeling.
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.logger = logging.getLogger(__name__)
    
    def convert(
        self,
        df: pd.DataFrame,
        x: Union[float, str],
        y: Union[float, str],
        z: str = 'DEPTH',
        properties: Optional[List[str]] = None,
        well_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert well log DataFrame to spatial format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Well log DataFrame with depth-based measurements
        x : float or str
            X coordinate (UTM or local) or column name containing x coordinates
        y : float or str
            Y coordinate (UTM or local) or column name containing y coordinates
        z : str, default 'DEPTH'
            Column name for depth/vertical coordinate
        properties : List[str], optional
            List of property columns to include. If None, includes all numeric columns
            except coordinates.
        well_name : str, optional
            Name/identifier for the well
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: X, Y, Z, and property columns
        """
        if df.empty:
            raise ValueError("DataFrame must not be empty")
        
        if z not in df.columns:
            raise ValueError(f"Depth column '{z}' not found in DataFrame")
        
        # Handle x and y coordinates
        if isinstance(x, str):
            if x not in df.columns:
                raise ValueError(f"X coordinate column '{x}' not found")
            x_coords = df[x].values
        else:
            x_coords = np.full(len(df), float(x))
        
        if isinstance(y, str):
            if y not in df.columns:
                raise ValueError(f"Y coordinate column '{y}' not found")
            y_coords = df[y].values
        else:
            y_coords = np.full(len(df), float(y))
        
        # Select properties to include
        if properties is None:
            # Include all numeric columns except coordinates
            exclude_cols = {z, x if isinstance(x, str) else None, 
                          y if isinstance(y, str) else None}
            exclude_cols.discard(None)
            properties = [col for col in df.columns 
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        # Create spatial DataFrame
        spatial_df = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'Z': df[z].values,
        })
        
        # Add properties
        for prop in properties:
            if prop in df.columns:
                spatial_df[prop] = df[prop].values
        
        # Add well identifier if provided
        if well_name:
            spatial_df['WELL'] = well_name
        
        self.logger.info(
            f"Converted {len(spatial_df)} samples to spatial format "
            f"with {len(properties)} properties"
        )
        
        return spatial_df
    
    def convert_multiple_wells(
        self,
        well_data: Dict[str, pd.DataFrame],
        coordinate_map: Dict[str, Tuple[float, float]],
        z: str = 'DEPTH',
        properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert multiple wells to a single spatial DataFrame.
        
        Parameters
        ----------
        well_data : Dict[str, pd.DataFrame]
            Dictionary mapping well names to DataFrames
        coordinate_map : Dict[str, Tuple[float, float]]
            Dictionary mapping well names to (x, y) coordinates
        z : str, default 'DEPTH'
            Column name for depth
        properties : List[str], optional
            Properties to include
            
        Returns
        -------
        pd.DataFrame
            Combined spatial DataFrame with all wells
        """
        spatial_dfs = []
        
        for well_name, df in well_data.items():
            if well_name not in coordinate_map:
                self.logger.warning(f"No coordinates for well {well_name}, skipping")
                continue
            
            x, y = coordinate_map[well_name]
            spatial_df = self.convert(df, x, y, z, properties, well_name)
            spatial_dfs.append(spatial_df)
        
        if not spatial_dfs:
            raise ValueError("No valid well data to convert")
        
        combined = pd.concat(spatial_dfs, ignore_index=True)
        self.logger.info(f"Combined {len(well_data)} wells into {len(combined)} spatial samples")
        
        return combined


class SpatialDataConverter:
    """
    Convert between different spatial data formats for reservoir modeling.
    """
    
    @staticmethod
    def to_pygeomodeling_format(
        df: pd.DataFrame,
        property_col: str,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'Z'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert spatial DataFrame to format expected by pygeomodeling.
        
        Parameters
        ----------
        df : pd.DataFrame
            Spatial DataFrame with X, Y, Z coordinates and properties
        property_col : str
            Column name of property to model
        x_col : str, default 'X'
            Column name for X coordinates
        y_col : str, default 'Y'
            Column name for Y coordinates
        z_col : str, default 'Z'
            Column name for Z coordinates
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (coordinates, values) where coordinates is (n_samples, 3)
            and values is (n_samples,)
        """
        if property_col not in df.columns:
            raise ValueError(f"Property column '{property_col}' not found")
        
        coords = df[[x_col, y_col, z_col]].values
        values = df[property_col].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        coords = coords[valid_mask]
        values = values[valid_mask]
        
        logger.debug(
            f"Converted {len(coords)} valid samples for property '{property_col}'"
        )
        
        return coords, values
    
    @staticmethod
    def create_prediction_grid(
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        nx: int = 50,
        ny: int = 50,
        nz: int = 20
    ) -> np.ndarray:
        """
        Create a 3D grid for spatial predictions.
        
        Parameters
        ----------
        x_range : Tuple[float, float]
            (x_min, x_max) range
        y_range : Tuple[float, float]
            (y_min, y_max) range
        z_range : Tuple[float, float]
            (z_min, z_max) range
        nx : int, default 50
            Number of points in X direction
        ny : int, default 50
            Number of points in Y direction
        nz : int, default 20
            Number of points in Z direction
            
        Returns
        -------
        np.ndarray
            Grid coordinates of shape (nx * ny * nz, 3)
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(z_range[0], z_range[1], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        logger.debug(f"Created prediction grid with {len(grid)} points")
        
        return grid

