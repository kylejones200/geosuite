#!/usr/bin/env python3
"""
Orebody Uncertainty Modeling Example

This example demonstrates geostatistical uncertainty quantification for orebody modeling:
1. Generate synthetic geochemical data (mimicking GA National Geochemical Survey)
2. Project to UTM coordinates
3. Log-transform values
4. Compute variogram and model spatial continuity
5. Perform Ordinary Kriging for baseline estimation
6. Generate Sequential Gaussian Simulations (SGS) for uncertainty
7. Compute uncertainty statistics (mean, std, percentiles)
8. Calculate exceedance probabilities

Based on Geoscience Australia geochemical data patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Import geosuite mining module
try:
    from geosuite.mining import (
        create_geochemical_dataframe,
        project_to_utm,
        compute_variogram,
        ordinary_kriging,
        sequential_gaussian_simulation,
        compute_simulation_statistics,
        compute_exceedance_probability,
        log_transform,
        exp_transform
    )
except ImportError as e:
    print(f"Error importing geosuite.mining: {e}")
    print("Make sure all optional dependencies are installed:")
    print("  pip install geopandas scikit-gstat pykrige")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_geochemical_data(
    n_samples: int = 2500,
    random_state: int = 123
) -> pd.DataFrame:
    """
    Generate synthetic geochemical data mimicking GA National Geochemical Survey patterns.
    
    Simulates orogenic gold deposits with high-grade shoots along NNW-trending structures.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sample_id, longitude, latitude, Au_ppm, sample_type
    """
    np.random.seed(random_state)
    
    # Yilgarn Craton bounding box (Western Australia)
    lon_min, lon_max = 119.0, 122.0
    lat_min, lat_max = -31.0, -28.5
    
    # Sample locations
    lons = np.random.uniform(lon_min, lon_max, n_samples)
    lats = np.random.uniform(lat_min, lat_max, n_samples)
    
    # Synthetic gold grades with spatial structure
    # Simulate orogenic gold deposits: high-grade shoots along NNW-trending structures
    center_lon, center_lat = 120.5, -29.75
    dist_to_center = np.sqrt((lons - center_lon)**2 + (lats - center_lat)**2)
    
    # Base grade pattern: exponential decay from center
    base_grade = 0.5 * np.exp(-dist_to_center / 0.5) + 0.05
    
    # Add structural control (NNW trend)
    nnw_component = (
        0.3 * np.exp(-((lons - center_lon) / 0.3)**2) *
        np.exp(-((lats - center_lat) / 1.0)**2)
    )
    
    # Combine with lognormal noise
    gold_ppm = (
        base_grade + nnw_component +
        np.random.lognormal(0, 0.5, n_samples) * 0.1
    )
    gold_ppm = np.clip(gold_ppm, 0.01, 50.0)  # Realistic range
    
    # Create dataframe
    geochem = pd.DataFrame({
        'sample_id': [f'GA{i:06d}' for i in range(n_samples)],
        'longitude': lons,
        'latitude': lats,
        'Au_ppm': gold_ppm,
        'sample_type': np.random.choice(
            ['soil', 'stream_sed', 'rock'],
            n_samples,
            p=[0.7, 0.2, 0.1]
        )
    })
    
    logger.info(f"✓ Generated {len(geochem):,} synthetic samples")
    logger.info(f"  Au grade range: {gold_ppm.min():.3f} - {gold_ppm.max():.3f} ppm")
    logger.info(f"  Au grade mean: {gold_ppm.mean():.3f} ppm")
    
    return geochem


def main():
    """Run complete orebody uncertainty modeling workflow."""
    
    logger.info("=" * 70)
    logger.info("Orebody Uncertainty Modeling Example")
    logger.info("=" * 70)
    
    # ============================================================================
    # 1. Generate synthetic geochemical data
    # ============================================================================
    logger.info("\n[1] Generating synthetic geochemical data...")
    geochem = generate_synthetic_geochemical_data(n_samples=2500, random_state=123)
    
    # ============================================================================
    # 2. Create GeoDataFrame and project to UTM
    # ============================================================================
    logger.info("\n[2] Creating GeoDataFrame and projecting to UTM...")
    gdf = create_geochemical_dataframe(
        geochem['longitude'].values,
        geochem['latitude'].values,
        geochem['Au_ppm'].values,
        sample_ids=geochem['sample_id'].values,
        value_name='Au_ppm',
        sample_type=geochem['sample_type'].values
    )
    
    # Project to UTM Zone 50S (Western Australia)
    gdf = project_to_utm(gdf, utm_zone=32750)
    logger.info(f"✓ Projected to UTM Zone 50S")
    logger.info(f"  X range: [{gdf['x'].min():.1f}, {gdf['x'].max():.1f}] m")
    logger.info(f"  Y range: [{gdf['y'].min():.1f}, {gdf['y'].max():.1f}] m")
    
    # ============================================================================
    # 3. Log-transform values
    # ============================================================================
    logger.info("\n[3] Log-transforming values...")
    gdf['log_Au'] = log_transform(gdf['Au_ppm'].values, add_one=True)
    logger.info(f"✓ Log-transformed")
    logger.info(f"  Log-Au range: [{gdf['log_Au'].min():.3f}, {gdf['log_Au'].max():.3f}]")
    logger.info(f"  Log-Au mean: {gdf['log_Au'].mean():.3f}")
    logger.info(f"  Log-Au std: {gdf['log_Au'].std():.3f}")
    
    # ============================================================================
    # 4. Compute variogram
    # ============================================================================
    logger.info("\n[4] Computing experimental variogram...")
    coords = gdf[['x', 'y']].values
    values = gdf['log_Au'].values
    
    V = compute_variogram(
        coords,
        values,
        model='spherical',
        maxlag=50000,  # 50 km max
        n_lags=20,
        normalize=False
    )
    
    variogram_params = {
        'sill': V.parameters[1],
        'range': V.parameters[0],
        'nugget': V.parameters[2]
    }
    
    logger.info(f"\nVariogram parameters:")
    logger.info(f"  Model: {V.model.__name__}")
    logger.info(f"  Range: {variogram_params['range']/1000:.1f} km")
    logger.info(f"  Sill: {variogram_params['sill']:.4f}")
    logger.info(f"  Nugget: {variogram_params['nugget']:.4f}")
    
    # ============================================================================
    # 5. Ordinary Kriging baseline
    # ============================================================================
    logger.info("\n[5] Performing Ordinary Kriging...")
    
    # Define estimation grid
    x_min, x_max = gdf['x'].quantile([0.05, 0.95])
    y_min, y_max = gdf['y'].quantile([0.05, 0.95])
    
    nx, ny = 100, 100
    gridx = np.linspace(x_min, x_max, nx)
    gridy = np.linspace(y_min, y_max, ny)
    
    logger.info(f"  Grid dimensions: {nx} × {ny} = {nx*ny:,} blocks")
    
    z_ok, ss_ok = ordinary_kriging(
        coords,
        values,
        gridx,
        gridy,
        variogram_model='spherical',
        variogram_parameters=variogram_params,
        verbose=False,
        enable_plotting=False
    )
    
    # Back-transform to ppm
    grade_ok = exp_transform(z_ok, subtract_one=True)
    
    logger.info(f"✓ Kriging complete")
    logger.info(f"  Grade range: {grade_ok.min():.3f} - {grade_ok.max():.3f} ppm")
    logger.info(f"  Grade mean: {grade_ok.mean():.3f} ppm")
    
    # ============================================================================
    # 6. Sequential Gaussian Simulation (SGS)
    # ============================================================================
    logger.info("\n[6] Generating Sequential Gaussian Simulations...")
    
    n_realizations = 50
    sim_stack = sequential_gaussian_simulation(
        coords,
        values,
        gridx,
        gridy,
        n_realizations=n_realizations,
        variogram_model='spherical',
        variogram_parameters=variogram_params,
        noise_level=0.1,
        random_state=42,
        verbose=True
    )
    
    logger.info(f"✓ Generated {n_realizations} realizations")
    logger.info(f"  Simulation stack shape: {sim_stack.shape}")
    
    # ============================================================================
    # 7. Compute uncertainty statistics
    # ============================================================================
    logger.info("\n[7] Computing uncertainty statistics...")
    
    stats = compute_simulation_statistics(
        sim_stack,
        transform_back=exp_transform  # Transform from log space to ppm
    )
    
    logger.info("\nSimulation Statistics (ppm):")
    logger.info(f"  Mean grade: {stats['mean'].mean():.3f} ppm")
    logger.info(f"  Std dev: {stats['std'].mean():.3f} ppm")
    logger.info(f"  P10 grade: {stats['p10'].mean():.3f} ppm")
    logger.info(f"  P50 grade: {stats['p50'].mean():.3f} ppm")
    logger.info(f"  P90 grade: {stats['p90'].mean():.3f} ppm")
    
    # ============================================================================
    # 8. Exceedance probability
    # ============================================================================
    logger.info("\n[8] Computing exceedance probabilities...")
    
    cutoff_ppm = 0.5  # Example economic cutoff (in original space)
    
    # Transform cutoff to log space for comparison with sim_stack
    def transform_cutoff_func(cutoff):
        return log_transform(np.array([cutoff]), add_one=True)[0]
    
    prob_exceed = compute_exceedance_probability(
        sim_stack,
        cutoff_ppm,
        transform_cutoff=transform_cutoff_func  # Transform cutoff from ppm to log space
    )
    
    logger.info(f"\nExceedance Probability (>{cutoff_ppm} ppm):")
    logger.info(f"  Mean probability: {prob_exceed.mean():.1%}")
    logger.info(f"  Max probability: {prob_exceed.max():.1%}")
    logger.info(f"  Blocks with >50% prob: {(prob_exceed > 0.5).sum()}/{prob_exceed.size}")
    logger.info(f"  Blocks with >90% prob: {(prob_exceed > 0.9).sum()}/{prob_exceed.size}")
    
    # ============================================================================
    # 9. Summary and visualization
    # ============================================================================
    logger.info("\n[9] Creating summary visualizations...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Orebody Uncertainty Modeling Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Sample locations and grades
    ax = axes[0, 0]
    scatter = ax.scatter(
        gdf['x'] / 1000, gdf['y'] / 1000,
        c=gdf['Au_ppm'], cmap='YlOrRd', s=20, alpha=0.6
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Sample Locations (Au ppm)')
    plt.colorbar(scatter, ax=ax, label='Au (ppm)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ordinary Kriging estimate
    ax = axes[0, 1]
    im = ax.imshow(
        grade_ok, extent=[x_min/1000, x_max/1000, y_min/1000, y_max/1000],
        origin='lower', cmap='YlOrRd', aspect='auto'
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Ordinary Kriging Estimate (Au ppm)')
    plt.colorbar(im, ax=ax, label='Au (ppm)')
    
    # Plot 3: Kriging variance (uncertainty)
    ax = axes[0, 2]
    im = ax.imshow(
        ss_ok, extent=[x_min/1000, x_max/1000, y_min/1000, y_max/1000],
        origin='lower', cmap='viridis', aspect='auto'
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Kriging Variance (Uncertainty)')
    plt.colorbar(im, ax=ax, label='Variance')
    
    # Plot 4: Mean of simulations
    ax = axes[1, 0]
    im = ax.imshow(
        stats['mean'], extent=[x_min/1000, x_max/1000, y_min/1000, y_max/1000],
        origin='lower', cmap='YlOrRd', aspect='auto'
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Mean of SGS Realizations (Au ppm)')
    plt.colorbar(im, ax=ax, label='Au (ppm)')
    
    # Plot 5: Standard deviation (uncertainty)
    ax = axes[1, 1]
    im = ax.imshow(
        stats['std'], extent=[x_min/1000, x_max/1000, y_min/1000, y_max/1000],
        origin='lower', cmap='plasma', aspect='auto'
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Standard Deviation (Uncertainty)')
    plt.colorbar(im, ax=ax, label='Std Dev (ppm)')
    
    # Plot 6: Exceedance probability
    ax = axes[1, 2]
    im = ax.imshow(
        prob_exceed, extent=[x_min/1000, x_max/1000, y_min/1000, y_max/1000],
        origin='lower', cmap='RdYlGn', aspect='auto', vmin=0, vmax=1
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(f'Exceedance Probability (>{cutoff_ppm} ppm)')
    plt.colorbar(im, ax=ax, label='Probability')
    
    plt.tight_layout()
    output_file = output_dir / 'orebody_uncertainty_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved visualization to {output_file}")
    plt.close()
    
    # ============================================================================
    # 10. Export results
    # ============================================================================
    logger.info("\n[10] Exporting results...")
    
    # Create results dataframe
    results = pd.DataFrame({
        'x_easting': np.tile(gridx, len(gridy)),
        'y_northing': np.repeat(gridy, len(gridx)),
        'grade_kriging': grade_ok.flatten(),
        'kriging_variance': ss_ok.flatten(),
        'grade_mean': stats['mean'].flatten(),
        'grade_std': stats['std'].flatten(),
        'grade_p10': stats['p10'].flatten(),
        'grade_p50': stats['p50'].flatten(),
        'grade_p90': stats['p90'].flatten(),
        'prob_exceed_0.5ppm': prob_exceed.flatten()
    })
    
    output_csv = output_dir / 'orebody_uncertainty_results.csv'
    results.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved results to {output_csv}")
    
    # Save variogram parameters
    variogram_output = output_dir / 'variogram_parameters.txt'
    with open(variogram_output, 'w') as f:
        f.write("Variogram Parameters\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: {V.model.__name__}\n")
        f.write(f"Range: {variogram_params['range']:.1f} m ({variogram_params['range']/1000:.2f} km)\n")
        f.write(f"Sill: {variogram_params['sill']:.4f}\n")
        f.write(f"Nugget: {variogram_params['nugget']:.4f}\n")
    logger.info(f"✓ Saved variogram parameters to {variogram_output}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ Orebody uncertainty modeling complete!")
    logger.info("=" * 70)
    logger.info(f"\nOutput files saved to: {output_dir.absolute()}")
    logger.info("  - orebody_uncertainty_results.png (visualizations)")
    logger.info("  - orebody_uncertainty_results.csv (results table)")
    logger.info("  - variogram_parameters.txt (variogram parameters)")


if __name__ == "__main__":
    main()

