"""
Demo data generator for testing geospatial features.

Creates synthetic Texas O&G data for prototyping.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def create_demo_texas_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create demo datasets for Texas oil & gas geospatial analysis.
    
    Returns:
        Tuple of (wells, pipelines, facilities, incidents) DataFrames
    """
    np.random.seed(42)
    
    # Texas bounding box (approx)
    tx_bounds = {
        'lon_min': -106.65, 'lon_max': -93.51,
        'lat_min': 25.84, 'lat_max': 36.50
    }
    
    # Focus on Permian Basin area (West Texas)
    permian_bounds = {
        'lon_min': -103.5, 'lon_max': -101.0,
        'lat_min': 31.0, 'lat_max': 33.0
    }
    
    # Generate wells (concentrated in Permian)
    n_wells = 1000
    wells = pd.DataFrame({
        'well_id': [f'API_{42000 + i:06d}' for i in range(n_wells)],
        'name': [f'Well_{i+1}' for i in range(n_wells)],
        'operator': np.random.choice(['Operator_A', 'Operator_B', 'Operator_C', 'Operator_D'], n_wells),
        'county': np.random.choice(['Midland', 'Ector', 'Loving', 'Reeves', 'Ward'], n_wells),
        'lon': np.random.uniform(permian_bounds['lon_min'], permian_bounds['lon_max'], n_wells),
        'lat': np.random.uniform(permian_bounds['lat_min'], permian_bounds['lat_max'], n_wells),
        'status': np.random.choice(['Active', 'Inactive', 'Plugged'], n_wells, p=[0.7, 0.2, 0.1]),
        'spud_date': pd.date_range('2015-01-01', '2024-12-31', periods=n_wells),
        'depth_ft': np.random.uniform(5000, 15000, n_wells).astype(int)
    })
    
    # Generate pipelines (linear features)
    n_pipelines = 50
    pipeline_segments = []
    for i in range(n_pipelines):
        # Each pipeline has 5-10 segments
        n_segments = np.random.randint(5, 11)
        start_lon = np.random.uniform(permian_bounds['lon_min'], permian_bounds['lon_max'])
        start_lat = np.random.uniform(permian_bounds['lat_min'], permian_bounds['lat_max'])
        
        for seg in range(n_segments):
            # Create line segments
            end_lon = start_lon + np.random.uniform(-0.1, 0.1)
            end_lat = start_lat + np.random.uniform(-0.1, 0.1)
            
            pipeline_segments.append({
                'pipeline_id': f'PIPE_{i+1}',
                'segment_id': f'PIPE_{i+1}_SEG_{seg+1}',
                'operator': np.random.choice(['Pipeline_Co_A', 'Pipeline_Co_B', 'Pipeline_Co_C']),
                'commodity': np.random.choice(['Crude Oil', 'Natural Gas', 'NGL']),
                'diameter_in': np.random.choice([6, 8, 12, 16, 20, 24]),
                'start_lon': start_lon,
                'start_lat': start_lat,
                'end_lon': end_lon,
                'end_lat': end_lat,
                'install_year': np.random.randint(1980, 2024)
            })
            
            start_lon, start_lat = end_lon, end_lat
    
    pipelines = pd.DataFrame(pipeline_segments)
    
    # Generate facilities (refineries, compressor stations)
    n_facilities = 20
    facilities = pd.DataFrame({
        'facility_id': [f'FAC_{i+1:03d}' for i in range(n_facilities)],
        'name': [f'Facility_{i+1}' for i in range(n_facilities)],
        'type': np.random.choice(['Refinery', 'Compressor Station', 'Processing Plant'], n_facilities),
        'operator': np.random.choice(['Operator_A', 'Operator_B', 'Operator_C'], n_facilities),
        'lon': np.random.uniform(permian_bounds['lon_min'], permian_bounds['lon_max'], n_facilities),
        'lat': np.random.uniform(permian_bounds['lat_min'], permian_bounds['lat_max'], n_facilities),
        'capacity': np.random.uniform(1000, 100000, n_facilities).astype(int),
        'status': np.random.choice(['Operating', 'Idle'], n_facilities, p=[0.9, 0.1])
    })
    
    # Generate incidents (PHMSA-style)
    n_incidents = 30
    incidents = pd.DataFrame({
        'incident_id': [f'INC_{2020000 + i}' for i in range(n_incidents)],
        'date': pd.date_range('2020-01-01', '2024-12-31', periods=n_incidents),
        'operator': np.random.choice(['Pipeline_Co_A', 'Pipeline_Co_B', 'Pipeline_Co_C'], n_incidents),
        'lon': np.random.uniform(permian_bounds['lon_min'], permian_bounds['lon_max'], n_incidents),
        'lat': np.random.uniform(permian_bounds['lat_min'], permian_bounds['lat_max'], n_incidents),
        'commodity': np.random.choice(['Crude Oil', 'Natural Gas', 'NGL'], n_incidents),
        'cause': np.random.choice(['Corrosion', 'Equipment Failure', 'Third Party', 'Natural Force'], n_incidents),
        'injury_count': np.random.poisson(0.1, n_incidents),
        'cost_usd': np.random.lognormal(10, 2, n_incidents).astype(int)
    })
    
    return wells, pipelines, facilities, incidents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    wells, pipelines, facilities, incidents = create_demo_texas_data()
    
    logger.info("Demo Texas O&G Data Generated")
    logger.info("=" * 60)
    logger.info(f"Wells: {len(wells)} records")
    logger.info(f"  Counties: {wells['county'].unique()}")
    logger.info(f"  Operators: {wells['operator'].nunique()}")
    logger.info(f"\nPipelines: {len(pipelines)} segments")
    logger.info(f"  Unique pipelines: {pipelines['pipeline_id'].nunique()}")
    logger.info(f"  Commodities: {pipelines['commodity'].unique()}")
    logger.info(f"\nFacilities: {len(facilities)} locations")
    logger.info(f"  Types: {facilities['type'].unique()}")
    logger.info(f"\nIncidents: {len(incidents)} events")
    logger.info(f"  Date range: {incidents['date'].min()} to {incidents['date'].max()}")
    logger.info("\nSample Well Data:")
    logger.info(wells.head(3))

