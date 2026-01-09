"""
Production data analysis and aggregation utilities.

This module provides functions for exploring and analyzing production data including:
- Temporal coverage analysis
- Spatial distribution analysis
- Field and pool aggregation
- Well-level statistics
- Production summary calculations
"""

from __future__ import annotations
from typing import Union, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def analyze_temporal_coverage(
    df: pd.DataFrame,
    date_col: str = 'ReportDate',
    well_id_col: Optional[str] = None
) -> Dict[str, any]:
    """
    Analyze temporal coverage of production data.
    
    Args:
        df: DataFrame with production data
        date_col: Name of date column
        well_id_col: Optional well identifier column
        
    Returns:
        Dictionary with temporal coverage statistics:
            - date_min: Minimum date
            - date_max: Maximum date
            - date_range_days: Total date range in days
            - unique_dates: Number of unique dates
            - date_gaps: List of gap periods
            - monthly_coverage: Count of records per month
            - active_wells_by_period: Number of active wells per period (if well_id_col provided)
            
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> coverage = analyze_temporal_coverage(df, date_col='ReportDate', well_id_col='API_WELLNO')
        >>> print(f"Date range: {coverage['date_min']} to {coverage['date_max']}")
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Convert to datetime if not already
    dates = pd.to_datetime(df[date_col], errors='coerce')
    dates = dates.dropna()
    
    if len(dates) == 0:
        raise ValueError("No valid dates found in date column")
    
    date_min = dates.min()
    date_max = dates.max()
    date_range_days = (date_max - date_min).days
    unique_dates = dates.nunique()
    
    # Monthly coverage
    monthly_coverage = dates.dt.to_period('M').value_counts().sort_index()
    
    # Detect gaps (months with no data)
    date_range = pd.period_range(
        start=date_min.to_period('M'),
        end=date_max.to_period('M'),
        freq='M'
    )
    gaps = date_range.difference(monthly_coverage.index)
    date_gaps = [str(gap) for gap in gaps] if len(gaps) > 0 else []
    
    result = {
        'date_min': date_min,
        'date_max': date_max,
        'date_range_days': date_range_days,
        'unique_dates': unique_dates,
        'date_gaps': date_gaps,
        'monthly_coverage': monthly_coverage.to_dict()
    }
    
    # Active wells by period if well_id_col provided
    if well_id_col and well_id_col in df.columns:
        monthly_active = df.groupby(
            dates.dt.to_period('M')
        )[well_id_col].nunique()
        result['active_wells_by_period'] = monthly_active.to_dict()
    
    return result


def analyze_spatial_distribution(
    df: pd.DataFrame,
    lat_col: str = 'Lat',
    lon_col: str = 'Long',
    production_col: Optional[str] = None
) -> Dict[str, any]:
    """
    Analyze spatial distribution of production data.
    
    Args:
        df: DataFrame with production data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        production_col: Optional production column for weighted statistics
        
    Returns:
        Dictionary with spatial statistics:
            - lat_min, lat_max: Latitude bounds
            - lon_min, lon_max: Longitude bounds
            - center_lat, center_lon: Centroid coordinates
            - spatial_bounds: Bounding box dictionary
            - point_count: Number of data points
            - unique_locations: Number of unique (lat, lon) pairs
            
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> spatial = analyze_spatial_distribution(df, lat_col='Lat', lon_col='Long', production_col='Oil')
        >>> print(f"Bounds: {spatial['lat_min']:.4f} to {spatial['lat_max']:.4f}")
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Latitude or longitude columns not found")
    
    # Drop rows with missing coordinates
    spatial_df = df[[lat_col, lon_col]].dropna()
    
    if len(spatial_df) == 0:
        raise ValueError("No valid spatial coordinates found")
    
    lat_min = spatial_df[lat_col].min()
    lat_max = spatial_df[lat_col].max()
    lon_min = spatial_df[lon_col].min()
    lon_max = spatial_df[lon_col].max()
    
    # Calculate centroid (weighted if production_col provided)
    if production_col and production_col in df.columns:
        # Weighted centroid
        weights = df[production_col].fillna(0).abs()
        center_lat = np.average(spatial_df[lat_col], weights=weights[:len(spatial_df)])
        center_lon = np.average(spatial_df[lon_col], weights=weights[:len(spatial_df)])
    else:
        # Simple centroid
        center_lat = spatial_df[lat_col].mean()
        center_lon = spatial_df[lon_col].mean()
    
    unique_locations = len(spatial_df.drop_duplicates([lat_col, lon_col]))
    
    return {
        'lat_min': float(lat_min),
        'lat_max': float(lat_max),
        'lon_min': float(lon_min),
        'lon_max': float(lon_max),
        'center_lat': float(center_lat),
        'center_lon': float(center_lon),
        'spatial_bounds': {
            'lat_min': float(lat_min),
            'lat_max': float(lat_max),
            'lon_min': float(lon_min),
            'lon_max': float(lon_max)
        },
        'point_count': len(spatial_df),
        'unique_locations': unique_locations
    }


def aggregate_by_field(
    df: pd.DataFrame,
    field_col: str = 'FieldName',
    production_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate production data by field.
    
    Args:
        df: DataFrame with production data
        field_col: Name of field column
        production_cols: List of production columns to aggregate (e.g., ['Oil', 'Gas', 'Wtr'])
        date_col: Optional date column for time-based aggregation
        
    Returns:
        DataFrame with field-level aggregations:
            - Total production (sum)
            - Average production (mean)
            - Maximum production (max)
            - Well count (nunique if well_id_col provided)
            - Date range (if date_col provided)
            
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> field_agg = aggregate_by_field(df, field_col='FieldName', 
        ...                                production_cols=['Oil', 'Gas'], 
        ...                                date_col='ReportDate')
    """
    if field_col not in df.columns:
        raise ValueError(f"Field column '{field_col}' not found")
    
    if production_cols is None:
        # Auto-detect production columns (common names)
        common_names = ['Oil', 'Gas', 'Wtr', 'Water', 'oil', 'gas', 'water']
        production_cols = [col for col in df.columns if col in common_names]
    
    if not production_cols:
        raise ValueError("No production columns specified or found")
    
    # Filter to available columns
    available_cols = [col for col in production_cols if col in df.columns]
    if not available_cols:
        raise ValueError("None of the specified production columns found in DataFrame")
    
    # Aggregate by field
    agg_dict = {}
    for col in available_cols:
        agg_dict[col] = ['sum', 'mean', 'max', 'count']
    
    field_agg = df.groupby(field_col).agg(agg_dict)
    
    # Flatten column names
    field_agg.columns = ['_'.join(col).strip() for col in field_agg.columns.values]
    
    # Add well count if available
    well_cols = ['API_WELLNO', 'WellName', 'UWI', 'well_id']
    well_col = next((col for col in well_cols if col in df.columns), None)
    if well_col:
        well_counts = df.groupby(field_col)[well_col].nunique()
        field_agg['well_count'] = well_counts
    
    # Add date range if date_col provided
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        date_ranges = df.groupby(field_col)[date_col].agg(['min', 'max'])
        field_agg['date_min'] = date_ranges['min']
        field_agg['date_max'] = date_ranges['max']
    
    return field_agg.sort_values(f'{available_cols[0]}_sum', ascending=False)


def aggregate_by_pool(
    df: pd.DataFrame,
    pool_col: str = 'Pool',
    production_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate production data by reservoir pool.
    
    Args:
        df: DataFrame with production data
        pool_col: Name of pool column
        production_cols: List of production columns to aggregate
        date_col: Optional date column for time-based aggregation
        
    Returns:
        DataFrame with pool-level aggregations (similar structure to aggregate_by_field)
        
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> pool_agg = aggregate_by_pool(df, pool_col='Pool', production_cols=['Oil', 'Gas'])
    """
    return aggregate_by_field(df, field_col=pool_col, production_cols=production_cols, date_col=date_col)


def calculate_well_statistics(
    df: pd.DataFrame,
    well_id_col: str = 'API_WELLNO',
    production_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate well-level production statistics.
    
    Args:
        df: DataFrame with production data
        well_id_col: Name of well identifier column
        production_cols: List of production columns
        date_col: Optional date column
        
    Returns:
        DataFrame with well-level statistics:
            - Total production (cumulative)
            - Average production (mean)
            - Maximum monthly production
            - Production record count
            - First/last production dates
            - Production span (days)
            
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> well_stats = calculate_well_statistics(df, well_id_col='API_WELLNO',
        ...                                        production_cols=['Oil', 'Gas'],
        ...                                        date_col='ReportDate')
    """
    if well_id_col not in df.columns:
        raise ValueError(f"Well ID column '{well_id_col}' not found")
    
    if production_cols is None:
        common_names = ['Oil', 'Gas', 'Wtr', 'Water', 'oil', 'gas', 'water']
        production_cols = [col for col in df.columns if col in common_names]
    
    if not production_cols:
        raise ValueError("No production columns specified or found")
    
    available_cols = [col for col in production_cols if col in df.columns]
    
    # Aggregate by well
    agg_dict = {}
    for col in available_cols:
        agg_dict[col] = ['sum', 'mean', 'max', 'count']
    
    well_stats = df.groupby(well_id_col).agg(agg_dict)
    well_stats.columns = ['_'.join(col).strip() for col in well_stats.columns.values]
    
    # Add date statistics if date_col provided
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        date_stats = df.groupby(well_id_col)[date_col].agg(['min', 'max', 'count'])
        well_stats['first_date'] = date_stats['min']
        well_stats['last_date'] = date_stats['max']
        well_stats['production_days'] = (
            pd.to_datetime(date_stats['max']) - pd.to_datetime(date_stats['min'])
        ).dt.days
        well_stats['record_count'] = date_stats['count']
    
    return well_stats.sort_values(f'{available_cols[0]}_sum', ascending=False)


def calculate_production_summary(
    df: pd.DataFrame,
    production_cols: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Calculate overall production summary statistics.
    
    Args:
        df: DataFrame with production data
        production_cols: List of production columns
        
    Returns:
        Dictionary with summary statistics for each production column:
            - mean, median, std, min, max
            - non_zero_count: Number of non-zero records
            - non_zero_pct: Percentage of non-zero records
            - total: Sum of all values
            
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> summary = calculate_production_summary(df, production_cols=['Oil', 'Gas', 'Wtr'])
        >>> print(f"Total Oil: {summary['Oil']['total']:,.0f} bbl")
    """
    if production_cols is None:
        common_names = ['Oil', 'Gas', 'Wtr', 'Water', 'oil', 'gas', 'water']
        production_cols = [col for col in df.columns if col in common_names]
    
    if not production_cols:
        raise ValueError("No production columns specified or found")
    
    available_cols = [col for col in production_cols if col in df.columns]
    
    summary = {}
    
    for col in available_cols:
        data = df[col].dropna()
        non_zero = (data > 0).sum()
        
        summary[col] = {
            'mean': float(data.mean()) if len(data) > 0 else 0.0,
            'median': float(data.median()) if len(data) > 0 else 0.0,
            'std': float(data.std()) if len(data) > 0 else 0.0,
            'min': float(data.min()) if len(data) > 0 else 0.0,
            'max': float(data.max()) if len(data) > 0 else 0.0,
            'total': float(data.sum()) if len(data) > 0 else 0.0,
            'non_zero_count': int(non_zero),
            'non_zero_pct': float(100 * non_zero / len(data)) if len(data) > 0 else 0.0,
            'count': len(data)
        }
    
    return summary


def aggregate_by_county(
    df: pd.DataFrame,
    county_col: str = 'County',
    production_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate production data by county.
    
    Args:
        df: DataFrame with production data
        county_col: Name of county column
        production_cols: List of production columns to aggregate
        date_col: Optional date column
        
    Returns:
        DataFrame with county-level aggregations (similar structure to aggregate_by_field)
        
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> county_agg = aggregate_by_county(df, county_col='County', production_cols=['Oil'])
    """
    return aggregate_by_field(df, field_col=county_col, production_cols=production_cols, date_col=date_col)


def calculate_production_density(
    df: pd.DataFrame,
    lat_col: str = 'Lat',
    lon_col: str = 'Long',
    production_col: str = 'Oil',
    well_id_col: Optional[str] = None,
    bin_size_degrees: float = 0.1
) -> pd.DataFrame:
    """
    Calculate production density on a spatial grid.
    
    Divides the spatial extent into bins and calculates production density
    (total production per unit area) for each bin. Useful for identifying
    production hotspots.
    
    Args:
        df: DataFrame with production data and coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        production_col: Name of production column
        well_id_col: Optional well identifier column (for well density)
        bin_size_degrees: Size of spatial bins in degrees (default: 0.1)
        
    Returns:
        DataFrame with columns:
            - lat_center, lon_center: Bin center coordinates
            - production_sum: Total production in bin
            - production_mean: Mean production per well/record in bin
            - well_count: Number of wells in bin (if well_id_col provided)
            - production_density: Production per unit area (production / bin_area)
            - well_density: Wells per unit area (if well_id_col provided)
            
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> density = calculate_production_density(
        ...     df,
        ...     lat_col='Lat',
        ...     lon_col='Long',
        ...     production_col='Oil',
        ...     well_id_col='API_WELLNO',
        ...     bin_size_degrees=0.1
        ... )
        >>> print(f"Highest density: {density.loc[density['production_density'].idxmax()]}")
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Latitude or longitude columns not found")
    
    if production_col not in df.columns:
        raise ValueError(f"Production column '{production_col}' not found")
    
    # Filter valid data
    df_clean = df[
        df[lat_col].notna() &
        df[lon_col].notna() &
        df[production_col].notna() &
        (df[production_col] > 0)
    ].copy()
    
    if len(df_clean) == 0:
        raise ValueError("No valid spatial production data found")
    
    # Create bins
    lat_min = df_clean[lat_col].min()
    lat_max = df_clean[lat_col].max()
    lon_min = df_clean[lon_col].min()
    lon_max = df_clean[lon_col].max()
    
    lat_bins = np.arange(lat_min, lat_max + bin_size_degrees, bin_size_degrees)
    lon_bins = np.arange(lon_min, lon_max + bin_size_degrees, bin_size_degrees)
    
    # Assign bins
    df_clean['lat_bin'] = pd.cut(df_clean[lat_col], bins=lat_bins, labels=False, include_lowest=True)
    df_clean['lon_bin'] = pd.cut(df_clean[lon_col], bins=lon_bins, labels=False, include_lowest=True)
    
    # Remove NaN bins
    df_clean = df_clean[df_clean['lat_bin'].notna() & df_clean['lon_bin'].notna()]
    
    # Aggregate by bin
    agg_dict = {
        production_col: ['sum', 'mean', 'count']
    }
    
    if well_id_col and well_id_col in df_clean.columns:
        agg_dict[well_id_col] = 'nunique'
    
    density_df = df_clean.groupby(['lat_bin', 'lon_bin']).agg(agg_dict)
    density_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                          for col in density_df.columns.values]
    
    # Calculate bin centers
    density_df = density_df.reset_index()
    density_df['lat_center'] = lat_bins[density_df['lat_bin'].astype(int)] + bin_size_degrees / 2
    density_df['lon_center'] = lon_bins[density_df['lon_bin'].astype(int)] + bin_size_degrees / 2
    
    # Calculate production density (production per unit area)
    # Approximate area of bin in square kilometers (accounting for latitude)
    # Area ≈ (bin_size_degrees * 111 km) * (bin_size_degrees * 111 km * cos(lat))
    mean_lat = df_clean[lat_col].mean()
    bin_area_km2 = (bin_size_degrees * 111.0) ** 2 * np.cos(np.radians(mean_lat))
    
    production_sum_col = f'{production_col}_sum'
    density_df['production_density'] = density_df[production_sum_col] / bin_area_km2
    
    # Calculate well density if well_id_col provided
    if well_id_col and well_id_col in df_clean.columns:
        well_count_col = f'{well_id_col}_nunique'
        density_df['well_density'] = density_df[well_count_col] / bin_area_km2
    
    # Clean up
    density_df = density_df.drop(columns=['lat_bin', 'lon_bin'])
    
    # Rename columns for clarity
    rename_dict = {
        production_sum_col: 'production_sum',
        f'{production_col}_mean': 'production_mean',
        f'{production_col}_count': 'record_count'
    }
    if well_id_col and well_id_col in df_clean.columns:
        rename_dict[f'{well_id_col}_nunique'] = 'well_count'
    
    density_df = density_df.rename(columns=rename_dict)
    
    return density_df.sort_values('production_density', ascending=False)


def identify_production_hotspots(
    df: pd.DataFrame,
    lat_col: str = 'Lat',
    lon_col: str = 'Long',
    production_col: str = 'Oil',
    well_id_col: Optional[str] = None,
    bin_size_degrees: float = 0.1,
    percentile_threshold: float = 90.0
) -> pd.DataFrame:
    """
    Identify production hotspots using density analysis.
    
    Calculates production density and identifies areas above a percentile
    threshold (e.g., top 10% of production density).
    
    Args:
        df: DataFrame with production data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        production_col: Name of production column
        well_id_col: Optional well identifier column
        bin_size_degrees: Size of spatial bins in degrees
        percentile_threshold: Percentile threshold for hotspot definition (default: 90)
        
    Returns:
        DataFrame with hotspot bins (top percentile by production density)
        
    Example:
        >>> df = pd.read_csv('production_data.csv')
        >>> hotspots = identify_production_hotspots(
        ...     df,
        ...     production_col='Oil',
        ...     percentile_threshold=95.0
        ... )
        >>> print(f"Found {len(hotspots)} hotspots")
    """
    density_df = calculate_production_density(
        df,
        lat_col=lat_col,
        lon_col=lon_col,
        production_col=production_col,
        well_id_col=well_id_col,
        bin_size_degrees=bin_size_degrees
    )
    
    # Identify hotspots (above percentile threshold)
    threshold = np.percentile(density_df['production_density'], percentile_threshold)
    hotspots = density_df[density_df['production_density'] >= threshold].copy()
    
    hotspots['percentile_threshold'] = percentile_threshold
    hotspots['threshold_value'] = threshold
    hotspots['is_hotspot'] = True
    
    logger.info(f"Identified {len(hotspots)} hotspots (top {100 - percentile_threshold:.1f}%)")
    logger.info(f"Density threshold: {threshold:.2f}")
    
    return hotspots.sort_values('production_density', ascending=False)

