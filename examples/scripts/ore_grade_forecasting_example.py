#!/usr/bin/env python3
"""
Ore Grade Forecasting Example

This example demonstrates multi-method ore grade forecasting:
1. Generate synthetic geochemical data (mimicking NGSA structure)
2. Prepare spatial features and coordinates
3. Create spatial cross-validation folds
4. Fit variogram for geostatistical baseline
5. Train Ordinary Kriging (OK) for baseline estimation
6. Train Gaussian Process Regression (GPR) with uncertainty
7. Train XGBoost for production forecasting
8. Generate grid predictions for all methods
9. Analyze uncertainty calibration
10. Compare methods

All methods use spatial cross-validation to prevent data leakage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Import geosuite mining module
try:
    from geosuite.mining import (
        generate_synthetic_geochemical_data,
        prepare_spatial_features,
        create_spatial_folds,
        fit_variogram_for_forecasting,
        ordinary_kriging_predict,
        train_gaussian_process,
        train_xgboost,
        create_prediction_grid,
        analyze_uncertainty_calibration,
        compare_forecasting_methods
    )
except ImportError as e:
    print(f"Error importing geosuite.mining: {e}")
    print("Make sure all optional dependencies are installed:")
    print("  pip install geopandas scikit-learn xgboost scikit-gstat pykrige scipy")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Complete ore grade forecasting pipeline."""
    logger.info('=' * 70)
    logger.info('ORE GRADE FORECASTING WITH GEOCHEMISTRY AND MACHINE LEARNING')
    logger.info('=' * 70)
    logger.info()
    
    # ============================================================================
    # 1. Generate synthetic geochemical data
    # ============================================================================
    logger.info('[1] Generating synthetic geochemical data...')
    df = generate_synthetic_geochemical_data(n_samples=250, random_state=42)
    
    # ============================================================================
    # 2. Prepare spatial features
    # ============================================================================
    logger.info('\n[2] Preparing spatial features...')
    gdf = prepare_spatial_features(df)
    
    # ============================================================================
    # 3. Create spatial cross-validation folds
    # ============================================================================
    logger.info('\n[3] Creating spatial cross-validation folds...')
    groups = create_spatial_folds(gdf, n_folds=5)
    
    # ============================================================================
    # 4. Fit variogram
    # ============================================================================
    logger.info('\n[4] Fitting variogram...')
    V = fit_variogram_for_forecasting(gdf)
    
    # ============================================================================
    # 5. Ordinary Kriging baseline
    # ============================================================================
    logger.info('\n[5] Performing Ordinary Kriging...')
    gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=100)
    
    # ============================================================================
    # 6. Train Gaussian Process Regression
    # ============================================================================
    logger.info('\n[6] Training Gaussian Process Regression...')
    gp_model, gp_pred, gp_std, gpr_metrics = train_gaussian_process(
        gdf, groups, random_state=42
    )
    
    # ============================================================================
    # 7. Train XGBoost
    # ============================================================================
    logger.info('\n[7] Training XGBoost...')
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(
        gdf, groups, random_state=42
    )
    
    # ============================================================================
    # 8. Generate grid predictions
    # ============================================================================
    logger.info('\n[8] Generating grid predictions...')
    grid_results = create_prediction_grid(
        gdf, gp_model, xgb_model, resolution=150
    )
    
    # ============================================================================
    # 9. Analyze uncertainty calibration
    # ============================================================================
    logger.info('\n[9] Analyzing uncertainty calibration...')
    calib_df = analyze_uncertainty_calibration(
        gdf['log_Au'].values, gp_pred, gp_std
    )
    
    # ============================================================================
    # 10. Compare methods
    # ============================================================================
    logger.info('\n[10] Comparing methods...')
    compare_forecasting_methods(gpr_metrics, xgb_metrics)
    
    # ============================================================================
    # 11. Create visualizations
    # ============================================================================
    logger.info('\n[11] Creating visualizations...')
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        'Ore Grade Forecasting: Multi-Method Comparison',
        fontsize=16,
        fontweight='bold'
    )
    
    # Plot 1: Sample locations and grades
    ax = axes[0, 0]
    scatter = ax.scatter(
        gdf['x'], gdf['y'],
        c=gdf['Au'], cmap='YlOrRd', s=30, alpha=0.7, edgecolors='k', linewidth=0.5
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Sample Locations (Au ppm)')
    plt.colorbar(scatter, ax=ax, label='Au (ppm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Ordinary Kriging
    ax = axes[0, 1]
    im = ax.imshow(
        ok_ppm,
        extent=[gdf['x'].min(), gdf['x'].max(), gdf['y'].min(), gdf['y'].max()],
        origin='lower', cmap='YlOrRd', aspect='auto'
    )
    ax.scatter(gdf['x'], gdf['y'], c='k', s=5, alpha=0.3)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Ordinary Kriging (Au ppm)')
    plt.colorbar(im, ax=ax, label='Au (ppm)')
    ax.set_aspect('equal')
    
    # Plot 3: Gaussian Process Regression
    ax = axes[0, 2]
    im = ax.imshow(
        grid_results['gp_mean'],
        extent=[gdf['x'].min(), gdf['x'].max(), gdf['y'].min(), gdf['y'].max()],
        origin='lower', cmap='YlOrRd', aspect='auto'
    )
    ax.scatter(gdf['x'], gdf['y'], c='k', s=5, alpha=0.3)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Gaussian Process Regression (Au ppm)')
    plt.colorbar(im, ax=ax, label='Au (ppm)')
    ax.set_aspect('equal')
    
    # Plot 4: XGBoost
    ax = axes[1, 0]
    im = ax.imshow(
        grid_results['xgb_pred'],
        extent=[gdf['x'].min(), gdf['x'].max(), gdf['y'].min(), gdf['y'].max()],
        origin='lower', cmap='YlOrRd', aspect='auto'
    )
    ax.scatter(gdf['x'], gdf['y'], c='k', s=5, alpha=0.3)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('XGBoost (Au ppm)')
    plt.colorbar(im, ax=ax, label='Au (ppm)')
    ax.set_aspect('equal')
    
    # Plot 5: GPR Uncertainty
    ax = axes[1, 1]
    im = ax.imshow(
        grid_results['gp_std'],
        extent=[gdf['x'].min(), gdf['x'].max(), gdf['y'].min(), gdf['y'].max()],
        origin='lower', cmap='plasma', aspect='auto'
    )
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('GPR Uncertainty (Std Dev)')
    plt.colorbar(im, ax=ax, label='Std Dev')
    ax.set_aspect('equal')
    
    # Plot 6: Uncertainty Calibration
    ax = axes[1, 2]
    ax.scatter(
        calib_df['predicted_std'],
        calib_df['actual_rmse'],
        s=calib_df['n_samples'] * 2,
        alpha=0.6,
        edgecolors='k',
        linewidth=0.5
    )
    # 1:1 line
    min_val = min(calib_df['predicted_std'].min(), calib_df['actual_rmse'].min())
    max_val = max(calib_df['predicted_std'].max(), calib_df['actual_rmse'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 line')
    ax.set_xlabel('Predicted Std Dev')
    ax.set_ylabel('Actual RMSE')
    ax.set_title('Uncertainty Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'ore_grade_forecasting_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved visualization to {output_file}")
    plt.close()
    
    # ============================================================================
    # 12. Export results
    # ============================================================================
    logger.info('\n[12] Exporting results...')
    
    # Create results dataframe for grid
    grid_flat = pd.DataFrame({
        'x_easting_km': grid_results['grid_x'].ravel(),
        'y_northing_km': grid_results['grid_y'].ravel(),
        'au_ok_ppm': grid_results['ok_mean'].ravel(),
        'au_ok_variance': grid_results['ok_var'].ravel(),
        'au_gpr_ppm': grid_results['gp_mean'].ravel(),
        'au_gpr_std': grid_results['gp_std'].ravel(),
        'au_xgb_ppm': grid_results['xgb_pred'].ravel()
    })
    
    output_csv = output_dir / 'ore_grade_forecasting_grid.csv'
    grid_flat.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved grid predictions to {output_csv}")
    
    # Create metrics summary
    metrics_summary = pd.DataFrame({
        'Method': ['Gaussian Process', 'XGBoost', 'Ordinary Kriging'],
        'MAE_log_ppm': [
            gpr_metrics['mae'],
            xgb_metrics['mae'],
            np.nan
        ],
        'RMSE_log_ppm': [
            gpr_metrics['rmse'],
            xgb_metrics['rmse'],
            np.nan
        ],
        '95pct_Coverage': [
            gpr_metrics['coverage'],
            np.nan,
            np.nan
        ],
        'Notes': [
            'Probabilistic with calibrated uncertainty',
            'Fast, production-ready',
            'Geostatistical baseline, no CV'
        ]
    })
    
    metrics_output = output_dir / 'forecasting_metrics_summary.csv'
    metrics_summary.to_csv(metrics_output, index=False)
    logger.info(f"✓ Saved metrics summary to {metrics_output}")
    
    logger.info('\n' + '=' * 70)
    logger.info('✓ Pipeline complete!')
    logger.info('=' * 70)
    logger.info(f'\nOutput files saved to: {output_dir.absolute()}')
    logger.info("  - ore_grade_forecasting_results.png (visualizations)")
    logger.info("  - ore_grade_forecasting_grid.csv (grid predictions)")
    logger.info("  - forecasting_metrics_summary.csv (metrics summary)")
    
    return {
        'data': gdf,
        'gp_model': gp_model,
        'xgb_model': xgb_model,
        'grid_results': grid_results,
        'metrics': {
            'gpr': gpr_metrics,
            'xgb': xgb_metrics
        },
        'calibration': calib_df
    }


if __name__ == '__main__':
    results = main()

