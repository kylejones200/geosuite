"""
Spatial property modelers using pygeomodeling for reservoir modeling.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for pygeomodeling
try:
    from pygeomodeling import UnifiedSPE9Toolkit
    from pygeomodeling.model_gp import StandardGP, DeepGP
    PYGEO_AVAILABLE = True
except ImportError:
    PYGEO_AVAILABLE = False
    UnifiedSPE9Toolkit = None
    StandardGP = None
    DeepGP = None


class SpatialPropertyModeler:
    """
    Model spatial reservoir properties using pygeomodeling.
    
    This class provides a high-level interface for modeling reservoir
    properties (permeability, porosity, etc.) spatially using Gaussian
    Process regression and Kriging.
    """
    
    def __init__(self, model_type: str = 'gpr', **model_kwargs):
        """
        Initialize the spatial property modeler.
        
        Parameters
        ----------
        model_type : str, default 'gpr'
            Type of model: 'gpr' (Gaussian Process using scikit-learn),
            'kriging' (requires pygeomodeling), or 'deep_gp' (requires pygeomodeling)
        **model_kwargs
            Additional arguments passed to the model
        """
        self.model_type = model_type
        
        # Check if advanced models require pygeomodeling
        if model_type in ['kriging', 'deep_gp'] and not PYGEO_AVAILABLE:
            raise ImportError(
                f"pygeomodeling is required for {model_type} models. "
                "Install with: pip install pygeomodeling or pip install geosuite[modeling]"
            )
        
        self.model_kwargs = model_kwargs
        self.model = None
        self.toolkit = None
        self.logger = logging.getLogger(__name__)
    
    def fit_permeability(
        self,
        spatial_data: pd.DataFrame,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'Z',
        property_col: str = 'PERMEABILITY'
    ) -> 'SpatialPropertyModeler':
        """
        Fit a spatial model for permeability.
        
        Parameters
        ----------
        spatial_data : pd.DataFrame
            Spatial DataFrame with coordinates and permeability values
        x_col : str, default 'X'
            Column name for X coordinates
        y_col : str, default 'Y'
            Column name for Y coordinates
        z_col : str, default 'Z'
            Column name for Z coordinates
        property_col : str, default 'PERMEABILITY'
            Column name for permeability values
            
        Returns
        -------
        SpatialPropertyModeler
            Self for method chaining
        """
        return self.fit_property(spatial_data, property_col, x_col, y_col, z_col)
    
    def fit_porosity(
        self,
        spatial_data: pd.DataFrame,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'Z',
        property_col: str = 'POROSITY'
    ) -> 'SpatialPropertyModeler':
        """
        Fit a spatial model for porosity.
        
        Parameters
        ----------
        spatial_data : pd.DataFrame
            Spatial DataFrame with coordinates and porosity values
        x_col : str, default 'X'
            Column name for X coordinates
        y_col : str, default 'Y'
            Column name for Y coordinates
        z_col : str, default 'Z'
            Column name for Z coordinates
        property_col : str, default 'POROSITY'
            Column name for porosity values
            
        Returns
        -------
        SpatialPropertyModeler
            Self for method chaining
        """
        return self.fit_property(spatial_data, property_col, x_col, y_col, z_col)
    
    def fit_property(
        self,
        spatial_data: pd.DataFrame,
        property_col: str,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'Z'
    ) -> 'SpatialPropertyModeler':
        """
        Fit a spatial model for any property.
        
        Parameters
        ----------
        spatial_data : pd.DataFrame
            Spatial DataFrame with coordinates and property values
        property_col : str
            Column name for property to model
        x_col : str, default 'X'
            Column name for X coordinates
        y_col : str, default 'Y'
            Column name for Y coordinates
        z_col : str, default 'Z'
            Column name for Z coordinates
            
        Returns
        -------
        SpatialPropertyModeler
            Self for method chaining
        """
        from .converters import SpatialDataConverter
        
        # Convert to pygeomodeling format
        coords, values = SpatialDataConverter.to_pygeomodeling_format(
            spatial_data, property_col, x_col, y_col, z_col
        )
        
        if len(coords) == 0:
            raise ValueError("No valid data points for modeling")
        
        # Create and train model
        if self.model_type == 'gpr':
            # Use scikit-learn GaussianProcessRegressor directly
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
            
            # Create combined kernel (RBF + Matern)
            kernel_type = self.model_kwargs.get('kernel_type', 'combined')
            if kernel_type == 'combined':
                kernel = ConstantKernel(1.0) * (RBF(length_scale=100.0) + Matern(length_scale=100.0, nu=1.5)) + WhiteKernel(noise_level=0.1)
            elif kernel_type == 'rbf':
                kernel = ConstantKernel(1.0) * RBF(length_scale=100.0) + WhiteKernel(noise_level=0.1)
            elif kernel_type == 'matern':
                kernel = ConstantKernel(1.0) * Matern(length_scale=100.0, nu=1.5) + WhiteKernel(noise_level=0.1)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=100.0) + WhiteKernel(noise_level=0.1)
            
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.model_kwargs.get('alpha', 1e-6),
                n_restarts_optimizer=self.model_kwargs.get('n_restarts', 10),
                normalize_y=True
            )
            
            # Fit model
            self.model.fit(coords, values)
            
            self.logger.info("Fitted scikit-learn GaussianProcessRegressor")
        elif self.model_type == 'kriging':
            # Use pykrige for Kriging (if available)
            try:
                from pykrige import OrdinaryKriging3D
                self.model = OrdinaryKriging3D(
                    coords[:, 0], coords[:, 1], coords[:, 2], values,
                    variogram_model=self.model_kwargs.get('variogram_model', 'linear')
                )
                self.logger.info("Fitted 3D Ordinary Kriging model")
            except ImportError:
                raise ImportError(
                    "pykrige is required for Kriging. Install with: pip install pykrige"
                )
        elif self.model_type == 'deep_gp':
            if not PYGEO_AVAILABLE:
                raise ImportError("pygeomodeling is required for Deep GP models")
            self.model = StandardGP(coords, values.reshape(-1, 1))
            self.model.train()
            self.logger.info("Fitted Deep GP model")
        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Supported types: 'gpr', 'kriging', 'deep_gp'"
            )
        
        self.logger.info(
            f"Fitted {self.model_type} model for {property_col} "
            f"with {len(coords)} training points"
        )
        
        return self
    
    def predict(
        self,
        coordinates: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict property values at given coordinates.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (n_points, 3) with (X, Y, Z) coordinates
        return_std : bool, default False
            If True, also return standard deviation (uncertainty)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Predicted values, or (predictions, std) if return_std=True
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.model_type == 'gpr':
            # scikit-learn GaussianProcessRegressor
            if return_std:
                pred, std = self.model.predict(coordinates, return_std=True)
                return pred.ravel(), std.ravel()
            else:
                return self.model.predict(coordinates).ravel()
        elif self.model_type == 'kriging':
            # Kriging prediction
            pred, var = self.model.execute(
                'points',
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2]
            )
            if return_std:
                std = np.sqrt(var)
                return pred, std
            return pred
        elif self.model_type == 'deep_gp':
            # Deep GP prediction
            try:
                import torch
            except ImportError:
                raise ImportError("torch is required for Deep GP models")
            
            self.model.eval()
            with torch.no_grad():
                pred_dist = self.model(coordinates)
                pred = pred_dist.mean.numpy()
                if return_std:
                    std = pred_dist.stddev.numpy()
                    return pred.ravel(), std.ravel()
                return pred.ravel()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


class ReservoirModelBuilder:
    """
    Build complete 3D reservoir models from well log data.
    """
    
    def __init__(self):
        """Initialize the reservoir model builder."""
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, SpatialPropertyModeler] = {}
    
    def add_property_model(
        self,
        name: str,
        spatial_data: pd.DataFrame,
        property_col: str,
        model_type: str = 'gpr',
        **kwargs
    ) -> 'ReservoirModelBuilder':
        """
        Add a property model to the reservoir model.
        
        Parameters
        ----------
        name : str
            Name of the property (e.g., 'permeability', 'porosity')
        spatial_data : pd.DataFrame
            Spatial DataFrame with coordinates and property values
        property_col : str
            Column name for the property
        model_type : str, default 'gpr'
            Type of model to use
        **kwargs
            Additional arguments for the model
            
        Returns
        -------
        ReservoirModelBuilder
            Self for method chaining
        """
        modeler = SpatialPropertyModeler(model_type=model_type, **kwargs)
        modeler.fit_property(spatial_data, property_col)
        self.models[name] = modeler
        
        self.logger.info(f"Added {model_type} model for property '{name}'")
        
        return self
    
    def predict_all_properties(
        self,
        coordinates: np.ndarray,
        return_std: bool = False
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Predict all properties at given coordinates.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (n_points, 3) with (X, Y, Z) coordinates
        return_std : bool, default False
            If True, also return standard deviations
            
        Returns
        -------
        Dict[str, np.ndarray or Tuple]
            Dictionary mapping property names to predictions
        """
        results = {}
        for name, modeler in self.models.items():
            results[name] = modeler.predict(coordinates, return_std=return_std)
        
        return results

