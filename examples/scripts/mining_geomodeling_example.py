#!/usr/bin/env python3
"""
Mining Geomodeling Example: Hybrid IDW + ML Block Model Generation

This example demonstrates the complete ore geomodeling workflow:
1. Process drillhole data (collar + assay)
2. Compute 3D coordinates
3. Build spatial features
4. Train hybrid IDW+ML model
5. Generate block model
6. Export results

Based on the NTGS drillhole database workflow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Import geosuite mining module
from geosuite.mining import (
    process_drillhole_data,
    merge_collar_assay,
    compute_3d_coordinates,
    idw_interpolate,
    build_spatial_features,
    create_block_model_grid,
    export_block_model,
    HybridOreModel,
    predict_block_grades
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_drillhole_data(n_holes=50, n_samples_per_hole=30, random_state=42):
    """
    Generate synthetic drillhole data for demonstration.
    
    In production, you would load real data from:
    - NTGS drillhole database
    - Company drillhole databases
    - CSV/Excel files with collar and assay data
    """
    rng = np.random.default_rng(random_state)
    
    # Synthetic collar data
    collars = []
    for hole in range(n_holes):
        # Random hole location in 2km x 2km area
        e_base = 400000 + rng.uniform(0, 2000)
        n_base = 7500000 + rng.uniform(0, 2000)
        rl_base = 300 + rng.normal(0, 10)
        
        collars.append({
            'HOLEID': f'DDH{hole:03d}',
            'EASTING': e_base,
            'NORTHING': n_base,
            'RL': rl_base
        })
    
    collar_df = pd.DataFrame(collars)
    
    # Synthetic assay data
    assays = []
    for hole in range(n_holes):
        hole_id = f'DDH{hole:03d}'
        e_base = collar_df.loc[hole, 'EASTING']
        n_base = collar_df.loc[hole, 'NORTHING']
        
        for sample in range(n_samples_per_hole):
            depth_from = sample * 5  # 5m intervals
            depth_to = depth_from + 5
            
            # Synthetic grade: function of location + depth + noise
            # High-grade zone in center
            dist_to_center = np.sqrt((e_base - 401000)**2 + (n_base - 7501000)**2)
            grade_base = 2.0 * np.exp(-dist_to_center / 500) + 0.3
            grade = grade_base * (1 + 0.3 * np.sin(depth_from / 20)) + rng.normal(0, 0.2)
            grade = np.clip(grade, 0.01, 10.0)
            
            assays.append({
                'HOLEID': hole_id,
                'FROM': depth_from,
                'TO': depth_to,
                'Au': grade  # Gold grade in g/t
            })
    
    assay_df = pd.DataFrame(assays)
    
    return collar_df, assay_df


def main():
    """Main workflow for ore geomodeling."""
    
    print("=" * 70)
    print("Mining Geomodeling Example: Hybrid IDW + ML Block Model")
    print("=" * 70)
    
    # ======================================================================
    # Step 1: Load/Generate Drillhole Data
    # ======================================================================
    print("\n[1/6] Loading drillhole data...")
    
    # For this example, generate synthetic data
    # In production, load from files:
    # collar_df = pd.read_csv('collar.csv')
    # assay_df = pd.read_csv('assay.csv')
    
    collar_df, assay_df = generate_synthetic_drillhole_data(
        n_holes=50, n_samples_per_hole=30, random_state=42
    )
    
    print(f"  Collar records: {len(collar_df):,}")
    print(f"  Assay records: {len(assay_df):,}")
    
    # ======================================================================
    # Step 2: Process and Merge Data
    # ======================================================================
    print("\n[2/6] Processing drillhole data...")
    
    # Auto-detect column names
    column_map = process_drillhole_data(collar_df, assay_df)
    
    # Merge collar and assay data
    samples = merge_collar_assay(collar_df, assay_df, column_map)
    
    # Compute 3D coordinates
    samples = compute_3d_coordinates(samples, assume_vertical=True)
    
    print(f"  ✓ Generated {len(samples):,} 3D sample points")
    print(f"  Grade range: {samples['grade'].min():.3f} - {samples['grade'].max():.3f} g/t")
    
    # ======================================================================
    # Step 3: Prepare Data for Modeling
    # ======================================================================
    print("\n[3/6] Preparing data for modeling...")
    
    # Extract coordinates and grades
    coords = samples[['x', 'y', 'z']].values
    grades = samples['grade'].values
    hole_ids = samples['hole_id'].values
    
    print(f"  Sample coordinates: {coords.shape}")
    print(f"  Grade statistics:")
    print(f"    Mean: {grades.mean():.3f} g/t")
    print(f"    Median: {grades.median():.3f} g/t")
    print(f"    Std: {grades.std():.3f} g/t")
    
    # ======================================================================
    # Step 4: Train Hybrid IDW+ML Model
    # ======================================================================
    print("\n[4/6] Training hybrid IDW+ML model...")
    
    model = HybridOreModel(
        idw_k=16,
        idw_power=2.0,
        ml_model_type='gradient_boosting',
        random_state=42
    )
    
    results = model.fit(
        coords=coords,
        grades=grades,
        group_ids=hole_ids,  # Spatial CV by drillhole
        cv_folds=5,
        compute_residuals=True
    )
    
    print(f"  ✓ Model trained")
    print(f"  Cross-validation MAE: {np.mean(results.cv_scores['mae']):.4f} ± "
          f"{np.std(results.cv_scores['mae']):.4f} g/t")
    print(f"  Cross-validation R²: {np.mean(results.cv_scores['r2']):.4f} ± "
          f"{np.std(results.cv_scores['r2']):.4f}")
    
    # ======================================================================
    # Step 5: Generate Block Model
    # ======================================================================
    print("\n[5/6] Generating block model...")
    
    # Create block model grid
    grid_coords, grid_info = create_block_model_grid(
        coords=coords,
        block_size_xy=25.0,  # 25m × 25m blocks
        block_size_z=10.0,   # 10m thick blocks
        quantile_padding=0.05
    )
    
    print(f"  Grid dimensions: {grid_info['nx']} × {grid_info['ny']} × {grid_info['nz']}")
    print(f"  Total blocks: {grid_info['n_blocks']:,}")
    
    # Predict block grades
    print("  Predicting block grades...")
    idw_grades, ml_residuals, final_grades = predict_block_grades(
        model=model,
        grid_coords=grid_coords,
        sample_coords=coords,
        sample_grades=grades
    )
    
    print(f"  ✓ Block model complete")
    print(f"  IDW grade range: {idw_grades.min():.4f} - {idw_grades.max():.4f} g/t")
    print(f"  Final grade range: {final_grades.min():.4f} - {final_grades.max():.4f} g/t")
    
    # ======================================================================
    # Step 6: Export and Visualize
    # ======================================================================
    print("\n[6/6] Exporting block model...")
    
    # Create block model DataFrame
    block_model = pd.DataFrame({
        'x_easting': grid_coords[:, 0],
        'y_northing': grid_coords[:, 1],
        'z_elevation': grid_coords[:, 2],
        'grade_idw': idw_grades,
        'grade_ml_fusion': final_grades,
        'ml_residual': ml_residuals
    })
    
    # Export to CSV (compatible with Vulcan, Datamine, Leapfrog, Surpac)
    output_dir = Path(__file__).parent.parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'block_model_mining_demo.csv'
    export_block_model(block_model, output_file, format='csv')
    
    print(f"  ✓ Block model exported to {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Block Model Summary Statistics")
    print("=" * 70)
    print(block_model[['grade_idw', 'grade_ml_fusion']].describe())
    
    # Create visualization (optional)
    print("\n  Creating visualization...")
    try:
        # Extract a horizontal slice at mid-depth
        z_mid = np.median(grid_coords[:, 2])
        z_tolerance = grid_info['block_size_z'] / 2
        slice_mask = np.abs(grid_coords[:, 2] - z_mid) < z_tolerance
        
        slice_data = block_model[slice_mask].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # IDW grades
        sc1 = axes[0].scatter(
            slice_data['x_easting'], slice_data['y_northing'],
            c=slice_data['grade_idw'], s=20, cmap='YlOrRd',
            vmin=0, vmax=3
        )
        axes[0].set_title('IDW Grade (Baseline)', fontsize=12)
        axes[0].set_xlabel('Easting (m)')
        axes[0].set_ylabel('Northing (m)')
        plt.colorbar(sc1, ax=axes[0], label='Grade (g/t Au)')
        
        # ML Fusion grades
        sc2 = axes[1].scatter(
            slice_data['x_easting'], slice_data['y_northing'],
            c=slice_data['grade_ml_fusion'], s=20, cmap='YlOrRd',
            vmin=0, vmax=3
        )
        axes[1].set_title('ML Fusion Grade', fontsize=12)
        axes[1].set_xlabel('Easting (m)')
        axes[1].set_ylabel('Northing (m)')
        plt.colorbar(sc2, ax=axes[1], label='Grade (g/t Au)')
        
        plt.tight_layout()
        
        plot_file = output_dir / 'grade_comparison_slice.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Visualization saved to {plot_file}")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Ore geomodeling workflow complete!")
    print("=" * 70)
    
    return block_model, model, results


if __name__ == '__main__':
    block_model, model, results = main()

