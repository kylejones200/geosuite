#!/usr/bin/env python3
"""
Interactive Mapping Example

This example demonstrates creating interactive Folium maps from:
1. Well location data
2. Kriging interpolation results
3. Combined datasets
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

try:
    from geosuite.plotting import (
        create_interactive_well_map,
        create_interactive_kriging_map,
        create_combined_map
    )
except ImportError as e:
    print(f"Error importing geosuite.plotting: {e}")
    print("Make sure optional dependencies are installed:")
    print("  pip install folium matplotlib")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_well_data(n_wells=100, random_state=42):
    """Generate synthetic well location and production data."""
    np.random.seed(random_state)
    
    # Western Pennsylvania region
    center_lat, center_lon = 40.5, -79.0
    
    # Generate well locations with spatial clustering
    cluster_centers = np.array([
        [center_lat + 0.5, center_lon - 0.5],
        [center_lat - 0.3, center_lon + 0.3],
        [center_lat + 0.2, center_lon + 0.2]
    ])
    
    wells = []
    for i in range(n_wells):
        cluster_idx = np.random.randint(0, len(cluster_centers))
        center = cluster_centers[cluster_idx]
        
        # Add random offset
        lat = center[0] + np.random.normal(0, 0.1)
        lon = center[1] + np.random.normal(0, 0.1)
        
        # Generate production (correlated with location)
        production = np.random.lognormal(5, 1.0) * (1.0 + np.exp(-((lat - center[0])**2 + (lon - center[1])**2) / 0.01))
        
        wells.append({
            'well_id': f'Well_{i:04d}',
            'latitude': lat,
            'longitude': lon,
            'production': production,
            'operator': np.random.choice(['Operator_A', 'Operator_B', 'Operator_C'])
        })
    
    df = pd.DataFrame(wells)
    
    logger.info(f"Generated {len(df)} wells")
    logger.info(f"Location range: Lat {df['latitude'].min():.4f}-{df['latitude'].max():.4f}, "
                f"Lon {df['longitude'].min():.4f}-{df['longitude'].max():.4f}")
    logger.info(f"Production range: {df['production'].min():.2f} - {df['production'].max():.2f}")
    
    return df


def generate_synthetic_kriging_data(df):
    """Generate synthetic kriging results from well data."""
    # Create grid
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    
    n_lat, n_lon = 50, 50
    grid_lat = np.linspace(lat_min, lat_max, n_lat)
    grid_lon = np.linspace(lon_min, lon_max, n_lon)
    grid_lat_2d, grid_lon_2d = np.meshgrid(grid_lat, grid_lon)
    
    # Simple interpolation (distance-weighted average)
    coordinates = df[['longitude', 'latitude']].values
    values = df['production'].values
    
    interpolated = np.zeros_like(grid_lat_2d)
    for i in range(n_lat):
        for j in range(n_lon):
            lat, lon = grid_lat[i], grid_lon[j]
            distances = np.sqrt((coordinates[:, 0] - lon)**2 + (coordinates[:, 1] - lat)**2)
            weights = 1.0 / (distances + 1e-6)**2
            weights /= weights.sum()
            interpolated[j, i] = np.sum(values * weights)
    
    # Variance (simplified)
    variance = np.random.rand(*interpolated.shape) * 100
    
    return {
        'grid_lon': grid_lon,
        'grid_lat': grid_lat,
        'interpolated': interpolated,
        'variance': variance,
        'coordinates': coordinates,
        'values': values
    }


def main():
    """Run interactive mapping example."""
    logger.info('=' * 70)
    logger.info('INTERACTIVE MAPPING EXAMPLE')
    logger.info('=' * 70)
    logger.info()
    
    # ============================================================================
    # 1. Generate synthetic well data
    # ============================================================================
    logger.info('[1] Generating synthetic well data...')
    df = generate_synthetic_well_data(n_wells=100, random_state=42)
    
    # ============================================================================
    # 2. Create interactive well map
    # ============================================================================
    logger.info('\n[2] Creating interactive well map...')
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    well_map = create_interactive_well_map(
        df,
        lat_col='latitude',
        lon_col='longitude',
        value_col='production',
        well_id_col='well_id',
        output_file=output_dir / 'wells_map.html',
        title='Production Well Locations',
        zoom_start=9
    )
    
    if well_map:
        logger.info("✓ Well map created successfully")
    else:
        logger.warning("Well map creation failed (folium may not be available)")
    
    # ============================================================================
    # 3. Generate kriging data and create kriging map
    # ============================================================================
    logger.info('\n[3] Creating interactive kriging map...')
    kriging_data = generate_synthetic_kriging_data(df)
    
    kriging_map = create_interactive_kriging_map(
        kriging_data['grid_lon'],
        kriging_data['grid_lat'],
        kriging_data['interpolated'],
        coordinates=kriging_data['coordinates'],
        values=kriging_data['values'],
        variance=kriging_data['variance'],
        output_file=output_dir / 'kriging_map.html',
        title='Kriging Interpolation Surface',
        production_type='Production',
        sample_step=3,
        max_wells=100,
        zoom_start=9
    )
    
    if kriging_map:
        logger.info("✓ Kriging map created successfully")
    else:
        logger.warning("Kriging map creation failed (folium may not be available)")
    
    # ============================================================================
    # 4. Create combined map with multiple datasets
    # ============================================================================
    logger.info('\n[4] Creating combined map...')
    
    # Create two datasets (oil and gas)
    df_oil = df.copy()
    df_oil['production'] = df['production'] * 6.0  # Convert to oil equivalent
    
    df_gas = df.copy()
    df_gas['production'] = df['production'] * 0.8  # Gas production
    
    # Prepare datasets for combined map
    datasets = [
        {
            'coordinates': df_oil[['longitude', 'latitude']].values,
            'values': df_oil['production'].values,
            'name': 'Oil Production',
            'color': 'blue',
            'max_samples': 100
        },
        {
            'coordinates': df_gas[['longitude', 'latitude']].values,
            'values': df_gas['production'].values,
            'name': 'Gas Production',
            'color': 'red',
            'max_samples': 100
        }
    ]
    
    combined_map = create_combined_map(
        datasets,
        output_file=output_dir / 'combined_map.html',
        zoom_start=9
    )
    
    if combined_map:
        logger.info("✓ Combined map created successfully")
    else:
        logger.warning("Combined map creation failed (folium may not be available)")
    
    # ============================================================================
    # 5. Summary
    # ============================================================================
    logger.info('\n' + '=' * 70)
    logger.info('✓ Interactive mapping example complete!')
    logger.info('=' * 70)
    logger.info(f'\nOutput files saved to: {output_dir.absolute()}')
    logger.info("  - wells_map.html (interactive well locations)")
    logger.info("  - kriging_map.html (kriging interpolation)")
    logger.info("  - combined_map.html (multiple datasets)")
    logger.info("\nTo view maps:")
    logger.info("  Open the HTML files in a web browser")
    logger.info("  Or use: python -m http.server (then navigate to http://localhost:8000)")
    
    return {
        'well_map': well_map,
        'kriging_map': kriging_map,
        'combined_map': combined_map
    }


if __name__ == '__main__':
    results = main()

