#!/usr/bin/env python3
"""
Forecast Validation Example

This example demonstrates forecast validation using cross-validation:
1. Generate synthetic production data for multiple wells
2. Perform cross-validation (holdout method)
3. Calculate forecast metrics (MSE, RMSE, MAE, MAPE)
4. Analyze summary statistics
5. Compare model types (exponential, hyperbolic, harmonic)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

try:
    from geosuite.forecasting import (
        calculate_forecast_metrics,
        cross_validate_well,
        evaluate_wells_dataset,
        calculate_summary_statistics,
        print_summary_statistics
    )
except ImportError as e:
    print(f"Error importing geosuite.forecasting: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install numpy pandas scikit-learn scipy")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_production_data(n_wells=50, n_months=36, random_state=42):
    """
    Generate synthetic production data for multiple wells.
    
    Simulates hyperbolic decline curves with noise.
    """
    np.random.seed(random_state)
    
    data = []
    
    for well_id in range(n_wells):
        # Random initial rate and decline parameters
        qi = np.random.uniform(100, 1000)  # Initial rate
        di = np.random.uniform(0.01, 0.1)  # Decline rate
        b = np.random.uniform(0.1, 0.9)   # Hyperbolic exponent
        
        # Generate time series
        dates = pd.date_range('2020-01-01', periods=n_months, freq='MS')
        
        # Hyperbolic decline with noise
        t = np.arange(n_months)
        rate = qi / np.power(1 + b * di * t, 1.0 / b)
        rate = rate * np.random.lognormal(0, 0.1, n_months)  # Add noise
        
        # Add some zeros (well shut-ins)
        zero_mask = np.random.rand(n_months) < 0.05  # 5% chance of zero
        rate[zero_mask] = 0
        
        # Create dataframe
        well_df = pd.DataFrame({
            'well_id': [f'Well_{well_id:03d}'] * n_months,
            'date': dates,
            'production': rate
        })
        
        data.append(well_df)
    
    df = pd.concat(data, ignore_index=True)
    
    logger.info(f"Generated {n_wells} wells with {n_months} months of data each")
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Production range: {df['production'].min():.2f} - {df['production'].max():.2f}")
    
    return df


def main():
    """Run forecast validation example."""
    logger.info('=' * 70)
    logger.info('FORECAST VALIDATION EXAMPLE')
    logger.info('=' * 70)
    logger.info()
    
    # ============================================================================
    # 1. Generate synthetic production data
    # ============================================================================
    logger.info('[1] Generating synthetic production data...')
    df = generate_synthetic_production_data(n_wells=50, n_months=36, random_state=42)
    
    # ============================================================================
    # 2. Evaluate wells using cross-validation
    # ============================================================================
    logger.info('\n[2] Performing cross-validation...')
    results = evaluate_wells_dataset(
        df,
        well_id_col='well_id',
        date_col='date',
        production_col='production',
        holdout_months=12,
        min_train_months=12,
        model_type='hyperbolic',
        sample_size=None,  # Use all wells
        random_state=42
    )
    
    logger.info(f"Successfully evaluated {len(results)} wells")
    
    # ============================================================================
    # 3. Calculate summary statistics
    # ============================================================================
    logger.info('\n[3] Calculating summary statistics...')
    summary = calculate_summary_statistics(results)
    print_summary_statistics(results)
    
    # ============================================================================
    # 4. Compare model types
    # ============================================================================
    logger.info('\n[4] Comparing model types...')
    model_types = ['exponential', 'hyperbolic', 'harmonic']
    model_results = {}
    
    for model_type in model_types:
        logger.info(f"\nEvaluating {model_type} model...")
        model_results[model_type] = evaluate_wells_dataset(
            df,
            well_id_col='well_id',
            date_col='date',
            production_col='production',
            holdout_months=12,
            min_train_months=12,
            model_type=model_type,
            sample_size=20,  # Sample for speed
            random_state=42
        )
        
        if len(model_results[model_type]) > 0:
            model_summary = calculate_summary_statistics(model_results[model_type])
            logger.info(f"\n{model_type.upper()} Model Performance:")
            if 'overall' in model_summary and 'rmse' in model_summary['overall']:
                rmse_stats = model_summary['overall']['rmse']
                logger.info(f"  Mean RMSE: {rmse_stats['mean']:.2f}")
                logger.info(f"  Median RMSE: {rmse_stats['median']:.2f}")
    
    # ============================================================================
    # 5. Create visualizations
    # ============================================================================
    logger.info('\n[5] Creating visualizations...')
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Distribution of forecast errors
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Forecast Validation Results', fontsize=14, fontweight='bold')
    
    # RMSE distribution
    ax = axes[0, 0]
    valid_rmse = results['rmse'].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(valid_rmse, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Frequency')
    ax.set_title('RMSE Distribution')
    ax.axvline(valid_rmse.mean(), color='r', linestyle='--', label=f'Mean: {valid_rmse.mean():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAPE distribution
    ax = axes[0, 1]
    valid_mape = results['mape'].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(valid_mape, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('MAPE (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('MAPE Distribution')
    ax.axvline(valid_mape.mean(), color='r', linestyle='--', label=f'Mean: {valid_mape.mean():.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE vs RMSE scatter
    ax = axes[1, 0]
    valid_mae = results['mae'].replace([np.inf, -np.inf], np.nan).dropna()
    common_idx = valid_rmse.index.intersection(valid_mae.index)
    ax.scatter(valid_rmse.loc[common_idx], valid_mae.loc[common_idx], alpha=0.6)
    ax.set_xlabel('RMSE')
    ax.set_ylabel('MAE')
    ax.set_title('MAE vs RMSE')
    ax.grid(True, alpha=0.3)
    
    # Model comparison
    ax = axes[1, 1]
    model_rmse_means = []
    model_names = []
    for model_type, model_res in model_results.items():
        if len(model_res) > 0:
            model_rmse = model_res['rmse'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(model_rmse) > 0:
                model_rmse_means.append(model_rmse.mean())
                model_names.append(model_type.title())
    
    if model_rmse_means:
        ax.bar(model_names, model_rmse_means, alpha=0.7, color=['blue', 'green', 'orange'])
        ax.set_ylabel('Mean RMSE')
        ax.set_title('Model Comparison')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'forecast_validation_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved visualization to {output_file}")
    plt.close()
    
    # ============================================================================
    # 6. Export results
    # ============================================================================
    logger.info('\n[6] Exporting results...')
    
    # Save main results
    output_csv = output_dir / 'forecast_validation_results.csv'
    results.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved results to {output_csv}")
    
    # Save model comparison
    if model_results:
        comparison_data = []
        for model_type, model_res in model_results.items():
            if len(model_res) > 0:
                model_rmse = model_res['rmse'].replace([np.inf, -np.inf], np.nan).dropna()
                model_mape = model_res['mape'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(model_rmse) > 0 and len(model_mape) > 0:
                    comparison_data.append({
                        'model_type': model_type,
                        'mean_rmse': model_rmse.mean(),
                        'median_rmse': model_rmse.median(),
                        'mean_mape': model_mape.mean(),
                        'median_mape': model_mape.median(),
                        'n_wells': len(model_res)
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_csv = output_dir / 'model_comparison.csv'
            comparison_df.to_csv(comparison_csv, index=False)
            logger.info(f"✓ Saved model comparison to {comparison_csv}")
    
    logger.info('\n' + '=' * 70)
    logger.info('✓ Forecast validation example complete!')
    logger.info('=' * 70)
    logger.info(f'\nOutput files saved to: {output_dir.absolute()}')
    logger.info("  - forecast_validation_results.png (visualizations)")
    logger.info("  - forecast_validation_results.csv (detailed results)")
    logger.info("  - model_comparison.csv (model comparison)")
    
    return {
        'results': results,
        'summary': summary,
        'model_results': model_results
    }


if __name__ == '__main__':
    results = main()

