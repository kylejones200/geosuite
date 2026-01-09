"""
RESQML (Reservoir Model XML) format support.

RESQML is an Energistics standard for reservoir modeling data exchange.
This module provides parsing and conversion utilities for RESQML v2.0+ files.
"""
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import RESQML libraries
try:
    import resqpy.model as rq
    import resqpy.organize as rqo
    import resqpy.grid as rqg
    import resqpy.property as rqp
    RESQPY_AVAILABLE = True
except ImportError:
    RESQPY_AVAILABLE = False
    logger.warning(
        "resqpy not available. RESQML support requires resqpy. "
        "Install with: pip install resqpy"
    )


class ResqmlParser:
    """
    Parser for RESQML reservoir modeling files.
    
    Supports reading grid geometries, properties, and well trajectories
    from RESQML v2.0+ files.
    
    Example:
        >>> parser = ResqmlParser()
        >>> grid_data = parser.load_grid('model.epc')
        >>> properties = parser.load_properties('model.epc', 'porosity')
    """
    
    def __init__(self):
        """Initialize RESQML parser."""
        if not RESQPY_AVAILABLE:
            raise ImportError(
                "resqpy is required for RESQML support. "
                "Install with: pip install resqpy"
            )
        self.model = None
    
    def load_model(self, epc_path: Union[str, Path]) -> None:
        """
        Load RESQML model from EPC file.
        
        Parameters
        ----------
        epc_path : str or Path
            Path to RESQML .epc file
        """
        epc_path = Path(epc_path)
        if not epc_path.exists():
            raise FileNotFoundError(f"RESQML file not found: {epc_path}")
        
        self.model = rq.Model(epc_file=str(epc_path))
        logger.info(f"Loaded RESQML model from {epc_path}")
    
    def load_grid(
        self,
        epc_path: Union[str, Path],
        grid_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load grid geometry from RESQML model.
        
        Parameters
        ----------
        epc_path : str or Path
            Path to RESQML .epc file
        grid_name : str, optional
            Name of grid to load (uses first grid if not specified)
            
        Returns
        -------
        dict
            Grid data including:
            - 'grid': Grid object
            - 'extent': Grid extent (nx, ny, nz)
            - 'origin': Grid origin coordinates
            - 'cell_centers': Cell center coordinates
        """
        if self.model is None:
            self.load_model(epc_path)
        
        # Get grid(s)
        grids = self.model.grids()
        if not grids:
            raise ValueError("No grids found in RESQML model")
        
        if grid_name:
            grid = next((g for g in grids if g.title == grid_name), None)
            if grid is None:
                raise ValueError(f"Grid '{grid_name}' not found")
        else:
            grid = grids[0]
        
        # Extract grid information
        extent = (grid.nk, grid.nj, grid.ni)  # (nz, ny, nx)
        
        # Get origin
        origin = grid.origin if hasattr(grid, 'origin') else None
        
        # Get cell centers (sample for large grids)
        try:
            centers = grid.centre_point()
            # For large grids, sample centers
            if centers.size > 1e6:
                sample_indices = np.random.choice(
                    centers.shape[0],
                    size=min(10000, centers.shape[0]),
                    replace=False
                )
                centers = centers[sample_indices]
        except Exception as e:
            logger.warning(f"Could not extract cell centers: {e}")
            centers = None
        
        return {
            'grid': grid,
            'extent': extent,
            'origin': origin,
            'cell_centers': centers,
            'title': grid.title
        }
    
    def load_properties(
        self,
        epc_path: Union[str, Path],
        property_name: Optional[str] = None,
        property_kind: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load property arrays from RESQML model.
        
        Parameters
        ----------
        epc_path : str or Path
            Path to RESQML .epc file
        property_name : str, optional
            Name of property to load
        property_kind : str, optional
            Kind of property (e.g., 'porosity', 'permeability', 'saturation')
            
        Returns
        -------
        dict
            Property data including:
            - 'values': Property values array
            - 'name': Property name
            - 'kind': Property kind
            - 'uom': Unit of measure
        """
        if self.model is None:
            self.load_model(epc_path)
        
        # Get properties
        properties = self.model.properties()
        if not properties:
            raise ValueError("No properties found in RESQML model")
        
        # Filter by name or kind
        filtered_props = properties
        if property_name:
            filtered_props = [p for p in filtered_props if p.title == property_name]
        if property_kind:
            filtered_props = [p for p in filtered_props 
                            if hasattr(p, 'property_kind') and 
                            p.property_kind == property_kind]
        
        if not filtered_props:
            raise ValueError(
                f"No properties found matching name={property_name}, "
                f"kind={property_kind}"
            )
        
        prop = filtered_props[0]
        
        # Get property values
        try:
            values = prop.array_ref()
        except Exception as e:
            logger.warning(f"Could not extract property values: {e}")
            values = None
        
        return {
            'values': values,
            'name': prop.title,
            'kind': getattr(prop, 'property_kind', None),
            'uom': getattr(prop, 'uom', None)
        }
    
    def load_well_trajectory(
        self,
        epc_path: Union[str, Path],
        well_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load well trajectory from RESQML model.
        
        Parameters
        ----------
        epc_path : str or Path
            Path to RESQML .epc file
        well_name : str, optional
            Name of well (uses first well if not specified)
            
        Returns
        -------
        pd.DataFrame
            Well trajectory with columns: MD, X, Y, Z
        """
        if self.model is None:
            self.load_model(epc_path)
        
        # Get wellbores
        wellbores = self.model.wellbores()
        if not wellbores:
            raise ValueError("No wellbores found in RESQML model")
        
        if well_name:
            wellbore = next((w for w in wellbores if w.title == well_name), None)
            if wellbore is None:
                raise ValueError(f"Well '{well_name}' not found")
        else:
            wellbore = wellbores[0]
        
        # Get trajectory
        try:
            trajectory = wellbore.trajectory()
            if trajectory is None:
                raise ValueError("No trajectory found for well")
            
            # Extract trajectory points
            md = trajectory.md()
            xyz = trajectory.xyz()
            
            df = pd.DataFrame({
                'MD': md,
                'X': xyz[:, 0],
                'Y': xyz[:, 1],
                'Z': xyz[:, 2]
            })
            
            return df
        except Exception as e:
            logger.error(f"Error loading well trajectory: {e}")
            raise


def load_resqml_grid(epc_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load RESQML grid.
    
    Parameters
    ----------
    epc_path : str or Path
        Path to RESQML .epc file
        
    Returns
    -------
    dict
        Grid data
    """
    parser = ResqmlParser()
    return parser.load_grid(epc_path)


def load_resqml_properties(
    epc_path: Union[str, Path],
    property_kind: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load RESQML properties.
    
    Parameters
    ----------
    epc_path : str or Path
        Path to RESQML .epc file
    property_kind : str, optional
        Kind of property to load
        
    Returns
    -------
    dict
        Property data
    """
    parser = ResqmlParser()
    return parser.load_properties(epc_path, property_kind=property_kind)


