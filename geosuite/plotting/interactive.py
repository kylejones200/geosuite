"""
Interactive mapping utilities using Folium.

This module provides utilities for creating interactive maps from
spatial analysis results, including kriging surfaces, well locations,
and production data visualization.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional dependency
try:
    import folium
    from folium import plugins
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning(
        "folium not available. Interactive maps require folium. "
        "Install with: pip install folium"
    )

try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning(
        "matplotlib not available. Color mapping requires matplotlib. "
        "Install with: pip install matplotlib"
    )


def create_interactive_kriging_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    interpolated: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    variance: Optional[np.ndarray] = None,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Kriging Interpolation Map",
    production_type: str = "Production",
    sample_step: int = 5,
    max_wells: int = 1000,
    zoom_start: int = 8
) -> Optional[folium.Map]:
    """
    Create interactive Folium map from kriging results.
    
    Args:
        grid_lon: Grid longitude coordinates (1D array)
        grid_lat: Grid latitude coordinates (1D array)
        interpolated: Interpolated values (2D array, shape: [lat, lon])
        coordinates: Optional well coordinates (n_wells, 2) - [lon, lat]
        values: Optional well production values (n_wells,)
        variance: Optional kriging variance (2D array, same shape as interpolated)
        output_file: Optional output file path for saving HTML
        title: Map title
        production_type: Production type label
        sample_step: Step size for sampling grid points (for performance)
        max_wells: Maximum number of wells to display
        zoom_start: Initial zoom level
        
    Returns:
        Folium map object, or None if folium not available
        
    Example:
        >>> grid_lon = np.linspace(-120, -118, 100)
        >>> grid_lat = np.linspace(38, 40, 100)
        >>> interpolated = np.random.rand(100, 100) * 1000
        >>> m = create_interactive_kriging_map(grid_lon, grid_lat, interpolated)
        >>> if m:
        ...     m.save('kriging_map.html')
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium not available, cannot create interactive map")
        return None
    
    grid_lon = np.asarray(grid_lon)
    grid_lat = np.asarray(grid_lat)
    interpolated = np.asarray(interpolated)
    
    # Calculate center of map
    center_lat = np.mean([grid_lat.min(), grid_lat.max()])
    center_lon = np.mean([grid_lon.min(), grid_lon.max()])
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery as alternative tile layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Sample grid points for performance
    grid_lon_sampled = grid_lon[::sample_step]
    grid_lat_sampled = grid_lat[::sample_step]
    
    # Handle 2D interpolated array
    if interpolated.ndim == 2:
        interpolated_sampled = interpolated[::sample_step, ::sample_step]
    else:
        logger.warning("Interpolated array must be 2D")
        interpolated_sampled = np.array([[0]])
    
    # Create contour-like visualization using circles
    valid_values = interpolated_sampled[~np.isnan(interpolated_sampled)]
    if len(valid_values) > 0 and MATPLOTLIB_AVAILABLE:
        vmin = float(valid_values.min())
        vmax = float(valid_values.max())
        
        # Create color scale (viridis-like)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('viridis')
        
        # Add interpolation surface as circles
        for i, lat in enumerate(grid_lat_sampled):
            for j, lon in enumerate(grid_lon_sampled):
                if i < interpolated_sampled.shape[0] and j < interpolated_sampled.shape[1]:
                    val = interpolated_sampled[i, j]
                    if not np.isnan(val):
                        color = mcolors.rgb2hex(cmap(norm(val))[:3])
                        folium.CircleMarker(
                            location=[float(lat), float(lon)],
                            radius=3,
                            popup=f'Interpolated: {val:.2f}',
                            tooltip=f'Value: {val:.2f}',
                            color=color,
                            fillColor=color,
                            fillOpacity=0.6,
                            weight=0
                        ).add_to(m)
    
    # Add well locations if provided
    if coordinates is not None and values is not None:
        coordinates = np.asarray(coordinates)
        values = np.asarray(values)
        
        # Sample wells for performance
        n_wells = min(max_wells, len(coordinates))
        if n_wells < len(coordinates):
            indices = np.random.choice(len(coordinates), n_wells, replace=False)
        else:
            indices = np.arange(len(coordinates))
        
        valid_values_wells = values[~np.isnan(values)]
        if len(valid_values_wells) > 0:
            vmin_wells = float(valid_values_wells.min())
            vmax_wells = float(valid_values_wells.max())
        else:
            vmin_wells = 0.0
            vmax_wells = 1.0
        
        well_data = []
        for idx in indices:
            if coordinates[idx].shape[0] >= 2:
                lon, lat = float(coordinates[idx, 0]), float(coordinates[idx, 1])
                val = float(values[idx]) if not np.isnan(values[idx]) else 0.0
                
                if not np.isnan(lon) and not np.isnan(lat):
                    # Color code by production value
                    val_norm = (val - vmin_wells) / (vmax_wells - vmin_wells + 1e-10)
                    
                    # Create color gradient
                    if val_norm < 0.33:
                        color = 'green'
                    elif val_norm < 0.67:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    well_data.append([lat, lon, val])
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        popup=f'Well Production: {val:.2f}',
                        tooltip=f'Production: {val:.2f}',
                        color='black',
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=1
                    ).add_to(m)
        
        # Add heatmap layer
        if len(well_data) > 0:
            HeatMap(well_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>{title}</h4>
    <p><b>Production Type:</b> {production_type}</p>
    <p><b>Wells Shown:</b> {n_wells if coordinates is not None else 0:,}</p>
    <p><b>Interpolation:</b> Kriging</p>
    <p style="font-size:10px">Green = Low<br>
    Orange = Medium<br>
    Red = High</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Interactive map saved to: {output_path}")
    
    return m


def create_interactive_well_map(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    value_col: Optional[str] = None,
    well_id_col: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Well Locations Map",
    zoom_start: int = 8,
    max_wells: int = 1000
) -> Optional[folium.Map]:
    """
    Create interactive map from well location data.
    
    Args:
        df: DataFrame with well location data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        value_col: Optional column name for production values
        well_id_col: Optional column name for well identifiers
        output_file: Optional output file path
        title: Map title
        zoom_start: Initial zoom level
        max_wells: Maximum number of wells to display
        
    Returns:
        Folium map object, or None if folium not available
        
    Example:
        >>> df = pd.DataFrame({
        ...     'latitude': [38.5, 39.0, 39.5],
        ...     'longitude': [-120.0, -119.5, -119.0],
        ...     'production': [1000, 2000, 1500]
        ... })
        >>> m = create_interactive_well_map(df, value_col='production')
        >>> if m:
        ...     m.save('wells_map.html')
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium not available, cannot create interactive map")
        return None
    
    # Filter valid data
    df_clean = df[
        df[lat_col].notna() &
        df[lon_col].notna()
    ].copy()
    
    if len(df_clean) == 0:
        logger.warning("No valid well locations found")
        return None
    
    # Sample if too many wells
    if len(df_clean) > max_wells:
        df_clean = df_clean.sample(n=max_wells, random_state=42)
        logger.info(f"Sampled {max_wells} wells for display")
    
    # Calculate center
    center_lat = df_clean[lat_col].mean()
    center_lon = df_clean[lon_col].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Prepare values for coloring
    if value_col and value_col in df_clean.columns:
        values = df_clean[value_col].values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            vmin = float(valid_values.min())
            vmax = float(valid_values.max())
        else:
            vmin = 0.0
            vmax = 1.0
    else:
        values = np.ones(len(df_clean))
        vmin = 0.0
        vmax = 1.0
    
    # Add well markers
    well_data = []
    for idx, row in df_clean.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        
        if value_col and value_col in row.index:
            val = float(row[value_col]) if pd.notna(row[value_col]) else 0.0
        else:
            val = 1.0
        
        well_id = str(row[well_id_col]) if well_id_col and well_id_col in row.index else f"Well {idx}"
        
        # Color code by value
        val_norm = (val - vmin) / (vmax - vmin + 1e-10)
        if val_norm < 0.33:
            color = 'green'
        elif val_norm < 0.67:
            color = 'orange'
        else:
            color = 'red'
        
        well_data.append([lat, lon, val])
        
        popup_text = f"Well: {well_id}"
        if value_col and value_col in row.index:
            popup_text += f"<br>Production: {val:.2f}"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=popup_text,
            tooltip=f"{well_id}: {val:.2f}" if value_col else well_id,
            color='black',
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    # Add heatmap if values provided
    if value_col and len(well_data) > 0:
        HeatMap(well_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>{title}</h4>
    <p><b>Wells Shown:</b> {len(df_clean):,}</p>
    <p style="font-size:10px">Green = Low<br>
    Orange = Medium<br>
    Red = High</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Interactive well map saved to: {output_path}")
    
    return m


def create_combined_map(
    datasets: List[Dict[str, Any]],
    output_file: Optional[Union[str, Path]] = None,
    zoom_start: int = 7
) -> Optional[folium.Map]:
    """
    Create a combined map with multiple layers.
    
    Args:
        datasets: List of dictionaries with keys:
            - 'coordinates': Well coordinates (n_wells, 2) - [lon, lat]
            - 'values': Production values (n_wells,)
            - 'name': Layer name
            - 'color': Optional marker color (default: blue, red, green, purple)
        output_file: Optional output file path
        zoom_start: Initial zoom level
        
    Returns:
        Folium map object, or None if folium not available
        
    Example:
        >>> datasets = [
        ...     {
        ...         'coordinates': np.array([[-120, 38], [-119, 39]]),
        ...         'values': np.array([1000, 2000]),
        ...         'name': 'Oil Production'
        ...     },
        ...     {
        ...         'coordinates': np.array([[-120.5, 38.5], [-119.5, 39.5]]),
        ...         'values': np.array([500, 1500]),
        ...         'name': 'Gas Production'
        ...     }
        ... ]
        >>> m = create_combined_map(datasets)
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium not available, cannot create interactive map")
        return None
    
    if len(datasets) == 0:
        logger.warning("No datasets provided")
        return None
    
    # Calculate center from first dataset
    coords = datasets[0].get('coordinates', np.array([[0, 0]]))
    center_lat = float(np.mean([coords[:, 1].min(), coords[:, 1].max()]) if coords.ndim == 2 else 0.0)
    center_lon = float(np.mean([coords[:, 0].min(), coords[:, 0].max()]) if coords.ndim == 2 else 0.0)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Default colors
    default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightred']
    
    # Add each dataset as a layer
    for i, dataset in enumerate(datasets):
        coordinates = np.asarray(dataset.get('coordinates', np.array([[0, 0]])))
        values = np.asarray(dataset.get('values', np.ones(len(coordinates))))
        name = dataset.get('name', f'Layer {i+1}')
        color = dataset.get('color', default_colors[i % len(default_colors)])
        max_samples = dataset.get('max_samples', 500)
        
        if len(coordinates) == 0:
            continue
        
        # Sample for performance
        n_samples = min(max_samples, len(coordinates))
        if n_samples < len(coordinates):
            indices = np.random.choice(len(coordinates), n_samples, replace=False)
        else:
            indices = np.arange(len(coordinates))
        
        feature_group = folium.FeatureGroup(name=name)
        
        for idx in indices:
            if coordinates[idx].shape[0] >= 2:
                lon = float(coordinates[idx, 0])
                lat = float(coordinates[idx, 1])
                val = float(values[idx]) if idx < len(values) and not np.isnan(values[idx]) else 0.0
                
                if not np.isnan(lon) and not np.isnan(lat):
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        popup=f'{name}: {val:.2f}',
                        tooltip=f'{name}: {val:.2f}',
                        color=color,
                        fillColor=color,
                        fillOpacity=0.6,
                        weight=1
                    ).add_to(feature_group)
        
        feature_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Combined map saved to: {output_path}")
    
    return m

