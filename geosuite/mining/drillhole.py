"""
Drillhole data processing utilities.

This module provides functions for processing drillhole collar and assay data,
including data loading, column detection, merging, and 3D coordinate computation.
"""

from __future__ import annotations
from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    case_sensitive: bool = False
) -> Optional[str]:
    """
    Find first matching column from candidate list.
    
    Searches for columns containing any of the candidate strings (case-insensitive
    by default). Useful for handling different naming conventions in drillhole data.
    
    Args:
        df: DataFrame to search
        candidates: List of candidate column name strings
        case_sensitive: If True, perform case-sensitive search
        
    Returns:
        First matching column name, or None if no match found
        
    Example:
        >>> df = pd.DataFrame({'HOLEID': [1, 2], 'EASTING': [100, 200]})
        >>> hole_col = find_column(df, ['HOLEID', 'HOLE_ID', 'DHID'])
        >>> print(hole_col)  # 'HOLEID'
    """
    for candidate in candidates:
        for col in df.columns:
            if case_sensitive:
                if candidate in col:
                    return col
            else:
                if candidate.lower() in col.lower():
                    return col
    return None


def process_drillhole_data(
    collar_df: pd.DataFrame,
    assay_df: pd.DataFrame,
    hole_id_candidates: List[str] = ['HOLEID', 'HOLE_ID', 'DHID'],
    easting_candidates: List[str] = ['EASTING', 'EAST', 'X', 'MGA_EASTING'],
    northing_candidates: List[str] = ['NORTHING', 'NORTH', 'Y', 'MGA_NORTHING'],
    rl_candidates: List[str] = ['RL', 'ELEVATION', 'Z'],
    depth_from_candidates: List[str] = ['FROM', 'DEPTH_FROM', 'FROM_M'],
    depth_to_candidates: List[str] = ['TO', 'DEPTH_TO', 'TO_M'],
    grade_candidates: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Automatically detect column names in drillhole collar and assay dataframes.
    
    Args:
        collar_df: DataFrame with collar (location) data
        assay_df: DataFrame with assay (grade) data
        hole_id_candidates: Candidates for hole ID column
        easting_candidates: Candidates for easting (X) column
        northing_candidates: Candidates for northing (Y) column
        rl_candidates: Candidates for RL (elevation) column
        depth_from_candidates: Candidates for depth from column
        depth_to_candidates: Candidates for depth to column
        grade_candidates: Optional list of grade column candidates.
                        If None, auto-detects first numeric column.
        
    Returns:
        Dictionary mapping standard names to detected column names:
            - 'hole_id_collar': Hole ID in collar data
            - 'hole_id_assay': Hole ID in assay data
            - 'easting': Easting coordinate
            - 'northing': Northing coordinate
            - 'rl': RL (elevation)
            - 'depth_from': Depth from
            - 'depth_to': Depth to
            - 'grade': Grade column
            
    Raises:
        ValueError: If required columns cannot be found
        
    Example:
        >>> collar = pd.read_csv('collar.csv')
        >>> assay = pd.read_csv('assay.csv')
        >>> columns = process_drillhole_data(collar, assay)
        >>> print(f"Hole ID column: {columns['hole_id_collar']}")
    """
    # Find collar columns
    hole_id_collar = find_column(collar_df, hole_id_candidates)
    easting = find_column(collar_df, easting_candidates)
    northing = find_column(collar_df, northing_candidates)
    rl = find_column(collar_df, rl_candidates)
    
    # Find assay columns
    hole_id_assay = find_column(assay_df, hole_id_candidates)
    depth_from = find_column(assay_df, depth_from_candidates)
    depth_to = find_column(assay_df, depth_to_candidates)
    
    # Find grade column
    if grade_candidates is None:
        grade_candidates = ['Au', 'AU', 'Au_ppm', 'Au_gpt', 'Gold', 'Cu', 'CU', 
                          'Cu_pct', 'Copper', 'Ag', 'Lead', 'Zn']
    
    grade_col = None
    for pref in grade_candidates:
        grade_col = find_column(assay_df, [pref])
        if grade_col:
            break
    
    # Fallback: first numeric column that's not a coordinate
    if grade_col is None:
        numeric_cols = assay_df.select_dtypes(include=[np.number]).columns
        exclude = ['x', 'y', 'z', 'from', 'to', 'depth', 'easting', 'northing', 'rl']
        grade_col = next((c for c in numeric_cols 
                         if c.lower() not in exclude), None)
    
    # Validate required columns
    required = {
        'hole_id_collar': hole_id_collar,
        'hole_id_assay': hole_id_assay,
        'easting': easting,
        'northing': northing,
        'rl': rl,
        'depth_from': depth_from,
        'depth_to': depth_to,
        'grade': grade_col
    }
    
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Could not find required columns: {missing}")
    
    logger.info(f"Detected columns: {required}")
    
    return required


def merge_collar_assay(
    collar_df: pd.DataFrame,
    assay_df: pd.DataFrame,
    column_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Merge collar and assay dataframes using detected column names.
    
    Args:
        collar_df: DataFrame with collar data
        assay_df: DataFrame with assay data
        column_map: Dictionary from process_drillhole_data()
        
    Returns:
        Merged DataFrame with standard column names:
            - hole_id: Drillhole identifier
            - from_m: Depth from (meters)
            - to_m: Depth to (meters)
            - grade: Grade value
            - easting: Easting coordinate
            - northing: Northing coordinate
            - rl: RL (elevation)
            
    Example:
        >>> columns = process_drillhole_data(collar, assay)
        >>> merged = merge_collar_assay(collar, assay, columns)
        >>> print(merged.head())
    """
    # Extract relevant columns
    assay_subset = assay_df[
        [column_map['hole_id_assay'], 
         column_map['depth_from'], 
         column_map['depth_to'], 
         column_map['grade']]
    ].copy()
    assay_subset.columns = ['hole_id', 'from_m', 'to_m', 'grade']
    assay_subset = assay_subset.dropna()
    
    collar_subset = collar_df[
        [column_map['hole_id_collar'], 
         column_map['easting'], 
         column_map['northing'], 
         column_map['rl']]
    ].copy()
    collar_subset.columns = ['hole_id', 'easting', 'northing', 'rl']
    
    # Merge
    samples = assay_subset.merge(collar_subset, on='hole_id', how='inner')
    
    return samples


def compute_3d_coordinates(
    df: pd.DataFrame,
    assume_vertical: bool = True,
    dip_col: Optional[str] = None,
    azimuth_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute 3D sample coordinates from drillhole data.
    
    Args:
        df: DataFrame with merged collar/assay data (from merge_collar_assay)
        assume_vertical: If True, assumes vertical drillholes
        dip_col: Optional column name for dip angle (degrees from horizontal)
        azimuth_col: Optional column name for azimuth angle (degrees)
        
    Returns:
        DataFrame with added columns:
            - depth_mid: Midpoint depth of assay interval
            - z: Z coordinate (RL - depth_mid for vertical holes)
            - x, y, z: 3D coordinates (for non-vertical holes)
            
    Example:
        >>> merged = merge_collar_assay(collar, assay, columns)
        >>> samples = compute_3d_coordinates(merged)
        >>> print(samples[['easting', 'northing', 'z', 'grade']].head())
    """
    df = df.copy()
    
    # Compute midpoint depth
    df['depth_mid'] = (df['from_m'] + df['to_m']) / 2.0
    
    if assume_vertical:
        # Vertical holes: Z = RL - depth
        df['z'] = df['rl'] - df['depth_mid']
        df['x'] = df['easting']
        df['y'] = df['northing']
    else:
        # Non-vertical holes: requires dip and azimuth
        if dip_col is None or azimuth_col is None:
            logger.warning("Non-vertical holes require dip and azimuth columns")
            # Fallback to vertical
            df['z'] = df['rl'] - df['depth_mid']
            df['x'] = df['easting']
            df['y'] = df['northing']
        else:
            # Convert dip/azimuth to 3D coordinates
            dip_rad = np.deg2rad(df[dip_col].values)
            azim_rad = np.deg2rad(df[azimuth_col].values)
            
            # Calculate offsets
            dx = df['depth_mid'] * np.sin(dip_rad) * np.cos(azim_rad)
            dy = df['depth_mid'] * np.sin(dip_rad) * np.sin(azim_rad)
            dz = -df['depth_mid'] * np.cos(dip_rad)  # Negative = down
            
            df['x'] = df['easting'] + dx
            df['y'] = df['northing'] + dy
            df['z'] = df['rl'] + dz
    
    # Clean data
    df = df[['hole_id', 'easting', 'northing', 'rl', 'x', 'y', 'z', 
             'from_m', 'to_m', 'depth_mid', 'grade']]
    df = df.dropna()
    df = df[np.isfinite(df['grade'])]
    
    return df

