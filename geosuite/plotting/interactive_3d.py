"""
Interactive 3D visualization for well logs and subsurface data.

Provides Plotly-based 3D viewers for well log visualization with
cross sections and multi-well correlation views.
"""
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning(
        "Plotly not available. Interactive 3D visualization requires Plotly. "
        "Install with: pip install plotly"
    )
    go = None
    px = None
    make_subplots = None


def create_3d_well_log_viewer(
    df: pd.DataFrame,
    depth_col: str = 'DEPTH',
    log_cols: Optional[List[str]] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    z_col: Optional[str] = None,
    facies_col: Optional[str] = None,
    well_name: Optional[str] = None,
    title: Optional[str] = None
) -> Any:
    """
    Create an interactive 3D well log viewer.
    
    Displays well logs in 3D space with depth as Z-axis and log values
    as X/Y axes or as color-coded traces along the wellbore.
    
    Parameters
    ----------
    df : pd.DataFrame
        Well log DataFrame
    depth_col : str, default 'DEPTH'
        Column name for depth
    log_cols : list of str, optional
        List of log columns to visualize. If None, uses common logs.
    x_col : str, optional
        Column name for X coordinate (if well has spatial coordinates)
    y_col : str, optional
        Column name for Y coordinate (if well has spatial coordinates)
    z_col : str, optional
        Column name for Z coordinate (alternative to depth_col)
    facies_col : str, optional
        Column name for facies (for color coding)
    well_name : str, optional
        Name of the well
    title : str, optional
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for 3D visualization. "
            "Install with: pip install plotly"
        )
    
    if log_cols is None:
        common_logs = ['GR', 'RHOB', 'NPHI', 'RT', 'PE']
        log_cols = [col for col in common_logs if col in df.columns][:3]
    
    if not log_cols:
        raise ValueError("No log columns found to visualize")
    
    # Use depth or z coordinate
    if z_col and z_col in df.columns:
        z_values = df[z_col].values
    else:
        z_values = df[depth_col].values
    
    # Get spatial coordinates if available
    has_spatial = x_col and y_col and x_col in df.columns and y_col in df.columns
    
    fig = go.Figure()
    
    # Create 3D visualization
    if has_spatial:
        # 3D wellbore trajectory with log values
        x_coords = df[x_col].values
        y_coords = df[y_col].values
        
        # Color by first log or facies
        if facies_col and facies_col in df.columns:
            color_values = df[facies_col].values
            colorbar_title = 'Facies'
        else:
            color_values = df[log_cols[0]].values
            colorbar_title = log_cols[0]
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_values,
            mode='markers+lines',
            marker=dict(
                size=3,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            line=dict(color='gray', width=2),
            name=well_name or 'Well',
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<extra></extra>',
            text=[f'{log_cols[0]}: {v:.2f}' for v in df[log_cols[0]].values]
        ))
    else:
        # 2D projection: depth vs log values
        for i, log_col in enumerate(log_cols):
            if log_col not in df.columns:
                continue
            
            log_values = df[log_col].values
            
            # Create 3D trace with offset for multiple logs
            x_offset = i * 50  # Offset each log track
            
            fig.add_trace(go.Scatter3d(
                x=[x_offset] * len(z_values),
                y=log_values,
                z=z_values,
                mode='lines+markers',
                name=log_col,
                line=dict(width=3),
                marker=dict(size=2),
                hovertemplate=f'<b>{log_col}</b><br>' +
                             f'Value: %{{y:.2f}}<br>' +
                             f'Depth: %{{z:.2f}}<br>' +
                             '<extra></extra>'
            ))
    
    # Update layout
    title_text = title or (f"3D Well Log Viewer - {well_name}" if well_name else "3D Well Log Viewer")
    
    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='X' if has_spatial else 'Log Track',
            yaxis_title='Y' if has_spatial else 'Log Value',
            zaxis_title='Depth (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1000,
        height=800
    )
    
    return fig


def create_multi_well_3d_viewer(
    wells_data: List[Dict[str, pd.DataFrame]],
    depth_col: str = 'DEPTH',
    x_col: str = 'X',
    y_col: str = 'Y',
    log_col: Optional[str] = None,
    facies_col: Optional[str] = None,
    title: str = 'Multi-Well 3D Viewer'
) -> Any:
    """
    Create an interactive 3D viewer for multiple wells.
    
    Displays multiple wells in 3D space with their trajectories
    and log values or facies color-coded.
    
    Parameters
    ----------
    wells_data : list of dict
        List of dictionaries, each containing:
        - 'df': DataFrame with well log data
        - 'name': Well name (optional)
        - 'x', 'y': Well coordinates (optional, if not in DataFrame)
    depth_col : str, default 'DEPTH'
        Column name for depth
    x_col : str, default 'X'
        Column name for X coordinate
    y_col : str, default 'Y'
        Column name for Y coordinate
    log_col : str, optional
        Log column to use for color coding
    facies_col : str, optional
        Facies column to use for color coding
    title : str, default 'Multi-Well 3D Viewer'
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for 3D visualization. "
            "Install with: pip install plotly"
        )
    
    fig = go.Figure()
    
    # Color scale for facies
    facies_colors = {
        'Sand': '#FFFF00',
        'Shale': '#808080',
        'Siltstone': '#C8C896',
        'Carbonate': '#0000FF',
        'Limestone': '#87CEEB',
        'Dolomite': '#FFB6C1',
        'Coal': '#000000',
        'Clean_Sand': '#FFFF00',
        'Shaly_Sand': '#F0E68C',
        'Mudstone': '#8B4513'
    }
    
    for well_idx, well_info in enumerate(wells_data):
        df = well_info['df']
        well_name = well_info.get('name', f'Well {well_idx + 1}')
        
        # Get coordinates
        if x_col in df.columns and y_col in df.columns:
            x_coords = df[x_col].values
            y_coords = df[y_col].values
        else:
            x_coords = np.full(len(df), well_info.get('x', well_idx * 100))
            y_coords = np.full(len(df), well_info.get('y', 0))
        
        z_values = df[depth_col].values
        
        # Determine color coding
        if facies_col and facies_col in df.columns:
            color_values = df[facies_col].values
            colors = [facies_colors.get(str(f), '#808080') for f in color_values]
            show_colorbar = False
        elif log_col and log_col in df.columns:
            color_values = df[log_col].values
            colors = color_values
            show_colorbar = True
        else:
            colors = f'rgb({well_idx * 50 % 255}, {well_idx * 100 % 255}, {well_idx * 150 % 255})'
            show_colorbar = False
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_values,
            mode='markers+lines',
            marker=dict(
                size=4,
                color=colors,
                showscale=show_colorbar,
                colorbar=dict(title=log_col) if show_colorbar else None,
                colorscale='Viridis' if show_colorbar else None
            ),
            line=dict(color=colors if isinstance(colors, str) else 'gray', width=3),
            name=well_name,
            hovertemplate=f'<b>{well_name}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Depth: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Depth (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=900
    )
    
    return fig


def create_cross_section_viewer(
    wells_data: List[Dict[str, pd.DataFrame]],
    section_azimuth: float = 0.0,
    depth_col: str = 'DEPTH',
    log_col: str = 'GR',
    facies_col: Optional[str] = None,
    title: str = 'Cross Section Viewer'
) -> Any:
    """
    Create an interactive cross section viewer.
    
    Displays a 2D cross section through multiple wells along a specified
    azimuth, showing log values or facies.
    
    Parameters
    ----------
    wells_data : list of dict
        List of dictionaries, each containing:
        - 'df': DataFrame with well log data
        - 'name': Well name (optional)
        - 'x', 'y': Well coordinates
    section_azimuth : float, default 0.0
        Azimuth of the cross section in degrees (0 = North)
    depth_col : str, default 'DEPTH'
        Column name for depth
    log_col : str, default 'GR'
        Log column to display
    facies_col : str, optional
        Facies column for color coding
    title : str, default 'Cross Section Viewer'
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for cross section visualization. "
            "Install with: pip install plotly"
        )
    
    fig = go.Figure()
    
    # Calculate distances along cross section
    azimuth_rad = np.radians(section_azimuth)
    
    for well_info in wells_data:
        df = well_info['df']
        well_name = well_info.get('name', 'Well')
        
        x_well = well_info.get('x', 0)
        y_well = well_info.get('y', 0)
        
        # Project well onto cross section line
        distance = x_well * np.cos(azimuth_rad) + y_well * np.sin(azimuth_rad)
        
        z_values = df[depth_col].values
        
        if log_col in df.columns:
            log_values = df[log_col].values
            
            # Color by facies if available
            if facies_col and facies_col in df.columns:
                colors = df[facies_col].values
                fig.add_trace(go.Scatter(
                    x=[distance] * len(z_values),
                    y=z_values,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Facies')
                    ),
                    name=well_name,
                    hovertemplate=f'<b>{well_name}</b><br>' +
                                 f'{log_col}: %{{text}}<br>' +
                                 'Depth: %{y:.2f}<br>' +
                                 '<extra></extra>',
                    text=log_values
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[distance] * len(z_values),
                    y=z_values,
                    mode='lines+markers',
                    marker=dict(
                        size=3,
                        color=log_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=log_col)
                    ),
                    name=well_name,
                    hovertemplate=f'<b>{well_name}</b><br>' +
                                 f'{log_col}: %{{text}}<br>' +
                                 'Depth: %{y:.2f}<br>' +
                                 '<extra></extra>',
                    text=log_values
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Distance along cross section (m)',
        yaxis_title='Depth (m)',
        yaxis=dict(autorange='reversed'),  # Invert depth axis
        width=1200,
        height=800
    )
    
    return fig

