"""
Coordinate Reference System (CRS) handling utilities.

Provides standardized CRS handling for geospatial data across GeoSuite.
Supports common CRS formats (EPSG, WKT, PROJ) and transformations.
"""
import logging
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import pyproj for CRS handling
try:
    from pyproj import CRS, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    CRS = None  # For type hints when pyproj is not available
    Transformer = None
    logger.warning(
        "pyproj not available. CRS support requires pyproj. "
        "Install with: pip install pyproj"
    )


class CRSHandler:
    """
    Handler for coordinate reference system operations.
    
    Provides standardized CRS handling, validation, and transformation
    for geospatial data in GeoSuite.
    
    Example:
        >>> handler = CRSHandler('EPSG:4326')
        >>> x, y = handler.transform(utm_x, utm_y, 'EPSG:32633')
    """
    
    def __init__(self, crs: Optional[Union[str, int, "CRS"]] = None):
        """
        Initialize CRS handler.
        
        Parameters
        ----------
        crs : str, int, or CRS, optional
            Coordinate reference system. Can be:
            - EPSG code (e.g., 'EPSG:4326' or 4326)
            - WKT string
            - PROJ string
            - pyproj.CRS object
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError(
                "pyproj is required for CRS support. "
                "Install with: pip install pyproj"
            )
        
        self.crs = None
        if crs is not None:
            self.set_crs(crs)
    
    def set_crs(self, crs: Union[str, int, "CRS"]) -> None:
        """
        Set coordinate reference system.
        
        Parameters
        ----------
        crs : str, int, or CRS
            Coordinate reference system
        """
        if isinstance(crs, int):
            crs = f'EPSG:{crs}'
        elif isinstance(crs, str) and not crs.startswith('EPSG:'):
            # Try to parse as EPSG code
            try:
                epsg_code = int(crs)
                crs = f'EPSG:{epsg_code}'
            except ValueError:
                pass  # Assume it's WKT or PROJ string
        
        self.crs = CRS.from_string(crs) if isinstance(crs, str) else crs
        logger.info(f"Set CRS: {self.crs}")
    
    def get_crs(self) -> Optional["CRS"]:
        """
        Get current CRS.
        
        Returns
        -------
        CRS or None
            Current coordinate reference system
        """
        return self.crs
    
    def get_epsg_code(self) -> Optional[int]:
        """
        Get EPSG code if available.
        
        Returns
        -------
        int or None
            EPSG code
        """
        if self.crs is None:
            return None
        
        try:
            return self.crs.to_epsg()
        except Exception:
            return None
    
    def transform(
        self,
        x: Union[float, np.ndarray, pd.Series],
        y: Union[float, np.ndarray, pd.Series],
        target_crs: Union[str, int, "CRS"],
        source_crs: Optional[Union[str, int, "CRS"]] = None
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Transform coordinates from source CRS to target CRS.
        
        Parameters
        ----------
        x : float, np.ndarray, or pd.Series
            X coordinates
        y : float, np.ndarray, or pd.Series
            Y coordinates
        target_crs : str, int, or CRS
            Target coordinate reference system
        source_crs : str, int, or CRS, optional
            Source coordinate reference system (uses instance CRS if not specified)
            
        Returns
        -------
        tuple
            Transformed (x, y) coordinates
        """
        if source_crs is None:
            if self.crs is None:
                raise ValueError("Source CRS must be specified")
            source_crs = self.crs
        else:
            if isinstance(source_crs, int):
                source_crs = f'EPSG:{source_crs}'
            elif isinstance(source_crs, str) and not source_crs.startswith('EPSG:'):
                try:
                    epsg_code = int(source_crs)
                    source_crs = f'EPSG:{epsg_code}'
                except ValueError:
                    pass
        
        if isinstance(target_crs, int):
            target_crs = f'EPSG:{target_crs}'
        elif isinstance(target_crs, str) and not target_crs.startswith('EPSG:'):
            try:
                epsg_code = int(target_crs)
                target_crs = f'EPSG:{epsg_code}'
            except ValueError:
                pass
        
        source_crs_obj = CRS.from_string(source_crs) if isinstance(source_crs, str) else source_crs
        target_crs_obj = CRS.from_string(target_crs) if isinstance(target_crs, str) else target_crs
        
        transformer = Transformer.from_crs(source_crs_obj, target_crs_obj, always_xy=True)
        
        # Convert to numpy arrays
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        
        # Transform
        x_transformed, y_transformed = transformer.transform(x_arr, y_arr)
        
        # Return in same format as input
        if isinstance(x, pd.Series):
            return pd.Series(x_transformed, index=x.index), pd.Series(y_transformed, index=y.index)
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x_transformed), float(y_transformed)
        else:
            return x_transformed, y_transformed
    
    def validate_coordinates(
        self,
        x: Union[float, np.ndarray, pd.Series],
        y: Union[float, np.ndarray, pd.Series],
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> bool:
        """
        Validate coordinates are within expected bounds.
        
        Parameters
        ----------
        x : float, np.ndarray, or pd.Series
            X coordinates
        y : float, np.ndarray, or pd.Series
            Y coordinates
        bounds : tuple, optional
            Expected bounds (x_min, y_min, x_max, y_max)
            
        Returns
        -------
        bool
            True if coordinates are valid
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        
        # Check for NaN or Inf
        if np.any(np.isnan(x_arr)) or np.any(np.isnan(y_arr)):
            return False
        if np.any(np.isinf(x_arr)) or np.any(np.isinf(y_arr)):
            return False
        
        # Check bounds if provided
        if bounds is not None:
            x_min, y_min, x_max, y_max = bounds
            if np.any(x_arr < x_min) or np.any(x_arr > x_max):
                return False
            if np.any(y_arr < y_min) or np.any(y_arr > y_max):
                return False
        
        return True


def standardize_crs(crs: Union[str, int, "CRS"]) -> "CRS":
    """
    Standardize CRS to pyproj.CRS object.
    
    Parameters
    ----------
    crs : str, int, or CRS
        Coordinate reference system
        
    Returns
    -------
    CRS
        Standardized CRS object
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError("pyproj is required for CRS support")
    
    if isinstance(crs, CRS):
        return crs
    
    if isinstance(crs, int):
        return CRS.from_epsg(crs)
    
    if isinstance(crs, str):
        if crs.startswith('EPSG:'):
            return CRS.from_epsg(int(crs.split(':')[1]))
        elif crs.isdigit():
            return CRS.from_epsg(int(crs))
        else:
            return CRS.from_string(crs)
    
    raise ValueError(f"Invalid CRS format: {crs}")


def get_common_crs() -> Dict[str, int]:
    """
    Get dictionary of common CRS codes.
    
    Returns
    -------
    dict
        Common CRS codes with descriptions
    """
    return {
        'WGS84': 4326,
        'UTM Zone 33N': 32633,
        'UTM Zone 33S': 32733,
        'UTM Zone 15N': 32615,
        'UTM Zone 15S': 32715,
        'NAD83': 4269,
        'NAD27': 4267,
        'Web Mercator': 3857
    }

