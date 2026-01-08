"""
Geospatial mapping and visualization for field-scale views.

Provides interactive maps using Folium and static maps using GeoPandas
for visualizing wells, facilities, and field boundaries.
"""
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Folium
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning(
        "Folium not available. Interactive maps require Folium. "
        "Install with: pip install folium"
    )
    folium = None
    plugins = None

# Try to import GeoPandas
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning(
        "GeoPandas not available. Static maps require GeoPandas. "
        "Install with: pip install geopandas"
    )
    gpd = None
    Point = None
    Polygon = None


def create_field_map(
    wells_df: pd.DataFrame,
    x_col: str = 'X',
    y_col: str = 'Y',
    well_name_col: Optional[str] = None,
    color_by: Optional[str] = None,
    title: str = 'Field Map',
    interactive: bool = True,
    basemap: str = 'OpenStreetMap'
) -> Any:
    """
    Create an interactive or static field map showing well locations.
    
    Parameters
    ----------
    wells_df : pd.DataFrame
        DataFrame with well coordinates and optional attributes
    x_col : str, default 'X'
        Column name for X coordinate (longitude or UTM X)
    y_col : str, default 'Y'
        Column name for Y coordinate (latitude or UTM Y)
    well_name_col : str, optional
        Column name for well names
    color_by : str, optional
        Column name to use for coloring wells
    title : str, default 'Field Map'
        Map title
    interactive : bool, default True
        If True, creates interactive Folium map. If False, creates static GeoPandas plot.
    basemap : str, default 'OpenStreetMap'
        Basemap style for Folium ('OpenStreetMap', 'Stamen Terrain', 'CartoDB positron')
        
    Returns
    -------
    folium.Map or matplotlib.figure.Figure
        Interactive map or static figure
    """
    if interactive:
        if not FOLIUM_AVAILABLE:
            raise ImportError(
                "Folium is required for interactive maps. "
                "Install with: pip install folium"
            )
        
        return _create_folium_map(
            wells_df, x_col, y_col, well_name_col, color_by, title, basemap
        )
    else:
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required for static maps. "
                "Install with: pip install geopandas"
            )
        
        return _create_geopandas_map(
            wells_df, x_col, y_col, well_name_col, color_by, title
        )


def _create_folium_map(
    wells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    well_name_col: Optional[str],
    color_by: Optional[str],
    title: str,
    basemap: str
) -> Any:
    """Create interactive Folium map."""
    # Determine center of map
    center_lat = wells_df[y_col].mean()
    center_lon = wells_df[x_col].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=basemap
    )
    
    # Add title
    title_html = f'<h3 align="center" style="font-size:20px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Color scheme
    try:
        import plotly.express as px
        has_plotly = True
    except ImportError:
        has_plotly = False
    
    if color_by and color_by in wells_df.columns:
        if pd.api.types.is_numeric_dtype(wells_df[color_by]):
            # Continuous color scale
            if has_plotly:
                colors = px.colors.sequential.Viridis
            else:
                colors = ['#440154', '#31688e', '#35b779', '#fde725']  # Viridis-like
            min_val = wells_df[color_by].min()
            max_val = wells_df[color_by].max()
            n_colors = len(colors)
        else:
            # Categorical colors
            unique_vals = wells_df[color_by].unique()
            if has_plotly:
                colors = px.colors.qualitative.Set3[:len(unique_vals)]
            else:
                colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
            color_map = dict(zip(unique_vals, colors[:len(unique_vals)]))
    else:
        colors = ['blue'] * len(wells_df)
    
    # Add well markers
    for idx, row in wells_df.iterrows():
        lat = row[y_col]
        lon = row[x_col]
        
        well_name = row[well_name_col] if well_name_col else f'Well {idx}'
        
        # Determine color
        if color_by and color_by in wells_df.columns:
            if pd.api.types.is_numeric_dtype(wells_df[color_by]):
                val = row[color_by]
                color_idx = int((val - min_val) / (max_val - min_val) * (n_colors - 1))
                color = colors[color_idx]
            else:
                color = color_map.get(row[color_by], 'blue')
        else:
            color = 'blue'
        
        # Create popup text
        popup_text = f'<b>{well_name}</b><br>'
        popup_text += f'Coordinates: ({lon:.2f}, {lat:.2f})<br>'
        
        if color_by:
            popup_text += f'{color_by}: {row[color_by]}<br>'
        
        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_text, max_width=200),
            color='black',
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add fullscreen and measure tools
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl().add_to(m)
    
    return m


def _create_geopandas_map(
    wells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    well_name_col: Optional[str],
    color_by: Optional[str],
    title: str
) -> Any:
    """Create static GeoPandas map."""
    import matplotlib.pyplot as plt
    import signalplot
    
    signalplot.apply()
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(wells_df[x_col], wells_df[y_col])]
    gdf = gpd.GeoDataFrame(wells_df, geometry=geometry, crs='EPSG:4326')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot wells
    if color_by and color_by in wells_df.columns:
        gdf.plot(
            ax=ax,
            column=color_by,
            marker='o',
            markersize=100,
            legend=True,
            legend_kwds={'label': color_by, 'shrink': 0.8},
            cmap='viridis' if pd.api.types.is_numeric_dtype(wells_df[color_by]) else 'Set3',
            edgecolor='black',
            linewidth=1.5
        )
    else:
        gdf.plot(
            ax=ax,
            marker='o',
            markersize=100,
            color='blue',
            edgecolor='black',
            linewidth=1.5
        )
    
    # Add well labels
    if well_name_col:
        for idx, row in gdf.iterrows():
            ax.annotate(
                text=row[well_name_col],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude / X', fontsize=12)
    ax.set_ylabel('Latitude / Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_well_trajectory_map(
    trajectories: List[Dict[str, Any]],
    title: str = 'Well Trajectories',
    interactive: bool = True
) -> Any:
    """
    Create a map showing well trajectories in 3D or 2D projection.
    
    Parameters
    ----------
    trajectories : list of dict
        List of trajectory dictionaries, each containing:
        - 'name': Well name
        - 'x', 'y', 'z': Arrays of coordinates
        - Optional: 'color', 'log_values', 'facies'
    title : str, default 'Well Trajectories'
        Map title
    interactive : bool, default True
        If True, creates interactive 3D Plotly plot. If False, creates 2D GeoPandas plot.
        
    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive 3D plot or static 2D figure
    """
    if interactive:
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive 3D trajectories. "
                "Install with: pip install plotly"
            )
        
        return _create_3d_trajectory_map(trajectories, title)
    else:
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required for static trajectory maps. "
                "Install with: pip install geopandas"
            )
        
        return _create_2d_trajectory_map(trajectories, title)


def _create_3d_trajectory_map(trajectories: List[Dict], title: str) -> Any:
    """Create interactive 3D trajectory map with Plotly."""
    try:
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for 3D trajectories")
    
    fig = go.Figure()
    
    for traj in trajectories:
        name = traj.get('name', 'Well')
        x = traj.get('x', [])
        y = traj.get('y', [])
        z = traj.get('z', [])
        color = traj.get('color', 'blue')
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=4),
            marker=dict(size=3, color=color),
            hovertemplate=f'<b>{name}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Depth (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200,
        height=900
    )
    
    return fig


def _create_2d_trajectory_map(trajectories: List[Dict], title: str) -> Any:
    """Create static 2D trajectory map with GeoPandas."""
    import matplotlib.pyplot as plt
    import signalplot
    
    signalplot.apply()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for traj in trajectories:
        name = traj.get('name', 'Well')
        x = traj.get('x', [])
        y = traj.get('y', [])
        color = traj.get('color', 'blue')
        
        ax.plot(x, y, marker='o', markersize=4, linewidth=2, label=name, color=color)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

