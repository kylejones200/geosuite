"""
Forecast validation and metrics calculation for decline curve analysis.

This module provides cross-validation utilities and forecast accuracy metrics
for evaluating decline curve forecast performance.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from .decline_models import (
    DeclineModel,
    HyperbolicDecline,
    ExponentialDecline,
    HarmonicDecline,
    fit_decline_model
)


def calculate_forecast_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics (MSE, RMSE, MAE, MAPE).
    
    Args:
        y_true: True production values
        y_pred: Predicted production values
        
    Returns:
        Dictionary with metrics: 'mse', 'rmse', 'mae', 'mape'
        Returns None for all metrics if no valid data points
        
    Example:
        >>> y_true = np.array([100, 90, 80, 70])
        >>> y_pred = np.array([95, 88, 82, 68])
        >>> metrics = calculate_forecast_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%")
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Remove NaN and infinite values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        logger.warning("No valid data points for metric calculation")
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'n_points': 0
        }
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE (avoid division by zero)
    mask_nonzero = y_true != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(
            np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])
        ) * 100
    else:
        mape = np.nan
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'n_points': len(y_true)
    }


def cross_validate_well(
    series: pd.Series,
    holdout_months: int = 12,
    min_train_months: int = 12,
    model_type: str = 'hyperbolic',
    resample_freq: str = 'MS'
) -> Optional[Dict[str, Any]]:
    """
    Cross-validate a single well by holding out last N months.
    
    Uses holdout validation: fits model on training data and evaluates
    on held-out test data.
    
    Args:
        series: Production time series (index should be datetime)
        holdout_months: Number of months to hold out for testing
        min_train_months: Minimum number of months required for training
        model_type: Decline model type ('hyperbolic', 'exponential', 'harmonic')
        resample_freq: Frequency for resampling ('MS' for monthly start)
        
    Returns:
        Dictionary with metrics and metadata, or None if insufficient data
        
    Example:
        >>> series = pd.Series([100, 90, 80, 70, 60, 50], 
        ...                    index=pd.date_range('2020-01', periods=6, freq='MS'))
        >>> results = cross_validate_well(series, holdout_months=2)
        >>> print(f"RMSE: {results['rmse']:.2f}")
    """
    # Ensure series is datetime-indexed
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            logger.warning("Series index must be datetime-like for cross-validation")
            return None
    
    # Resample if needed
    if resample_freq:
        series = series.resample(resample_freq).sum()
    
    # Remove zeros and invalid values
    series = series[series > 0]
    
    # Check sufficient data
    if len(series) < min_train_months + holdout_months:
        return None
    
    # Split data
    train_series = series.iloc[:-holdout_months]
    test_series = series.iloc[-holdout_months:]
    
    try:
        # Fit model on training data
        model = fit_decline_model(
            train_series.index,
            train_series.values,
            model_type=model_type
        )
        
        # Forecast for test period
        n_periods = len(test_series)
        last_train_time = (train_series.index[-1] - train_series.index[0]).days
        period_length = (test_series.index[1] - test_series.index[0]).days if len(test_series) > 1 else 30
        
        forecast_df = model.forecast(
            n_periods=n_periods,
            period_length=period_length / 30.0,  # Convert to months
            start_time=last_train_time / 30.0
        )
        
        # Align forecast with test series
        # Create forecast series with test index
        forecast_rates = forecast_df['rate'].values
        if len(forecast_rates) < len(test_series):
            # Pad with last value if needed
            forecast_rates = np.pad(
                forecast_rates,
                (0, len(test_series) - len(forecast_rates)),
                mode='edge'
            )
        elif len(forecast_rates) > len(test_series):
            forecast_rates = forecast_rates[:len(test_series)]
        
        forecast_series = pd.Series(forecast_rates, index=test_series.index)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(
            test_series.values,
            forecast_series.values
        )
        
        if metrics['n_points'] == 0:
            return None
        
        return {
            **metrics,
            'holdout_months': holdout_months,
            'train_months': len(train_series),
            'test_months': len(test_series),
            'model_type': model_type,
            'train_date_start': train_series.index[0],
            'train_date_end': train_series.index[-1],
            'test_date_start': test_series.index[0],
            'test_date_end': test_series.index[-1]
        }
        
    except Exception as e:
        logger.warning(f"Error in cross-validation: {e}")
        return None


def evaluate_wells_dataset(
    df: pd.DataFrame,
    well_id_col: str,
    date_col: str,
    production_col: str,
    holdout_months: int = 12,
    min_train_months: int = 12,
    model_type: str = 'hyperbolic',
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Evaluate multiple wells using cross-validation.
    
    Args:
        df: DataFrame with well production data
        well_id_col: Column name for well identifier
        date_col: Column name for date
        production_col: Column name for production values
        holdout_months: Number of months to hold out for testing
        min_train_months: Minimum months required for training
        model_type: Decline model type
        sample_size: Optional sample size for large datasets (None = all)
        random_state: Random seed for sampling
        
    Returns:
        DataFrame with cross-validation results for each well
        
    Example:
        >>> df = pd.DataFrame({
        ...     'well_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        ...     'date': pd.date_range('2020-01', periods=6, freq='MS'),
        ...     'production': [100, 90, 80, 200, 180, 160]
        ... })
        >>> results = evaluate_wells_dataset(df, 'well_id', 'date', 'production')
        >>> print(f"Evaluated {len(results)} wells")
    """
    # Prepare data
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    df_clean = df[
        df[date_col].notna() &
        df[production_col].notna() &
        (df[production_col] > 0)
    ].copy()
    
    if len(df_clean) == 0:
        logger.warning("No valid data for evaluation")
        return pd.DataFrame()
    
    # Get wells with sufficient data
    well_counts = df_clean[well_id_col].value_counts()
    required_points = min_train_months + holdout_months
    wells_with_data = well_counts[well_counts >= required_points].index.tolist()
    
    logger.info(f"Found {len(wells_with_data)} wells with >= {required_points} data points")
    
    # Sample if requested
    if sample_size is not None and len(wells_with_data) > sample_size:
        if random_state is not None:
            np.random.seed(random_state)
        wells_with_data = np.random.choice(wells_with_data, sample_size, replace=False).tolist()
        logger.info(f"Sampling {sample_size} wells for evaluation")
    
    # Evaluate each well
    results = []
    for well_id in wells_with_data:
        well_data = df_clean[df_clean[well_id_col] == well_id].copy()
        well_data = well_data.sort_values(date_col)
        
        # Create time series
        series = well_data.set_index(date_col)[production_col]
        series = series[series > 0]
        
        # Cross-validate
        metrics = cross_validate_well(
            series,
            holdout_months=holdout_months,
            min_train_months=min_train_months,
            model_type=model_type
        )
        
        if metrics:
            metrics[well_id_col] = well_id
            results.append(metrics)
    
    if len(results) == 0:
        logger.warning("No wells successfully evaluated")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    logger.info(f"Successfully evaluated {len(results_df)} wells")
    
    return results_df


def calculate_summary_statistics(
    results_df: pd.DataFrame,
    group_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate summary statistics across wells.
    
    Args:
        results_df: DataFrame with cross-validation results
        group_by: Optional column name to group by (e.g., 'state', 'model_type')
        
    Returns:
        Dictionary with summary statistics
        
    Example:
        >>> results_df = evaluate_wells_dataset(df, 'well_id', 'date', 'production')
        >>> summary = calculate_summary_statistics(results_df)
        >>> print(f"Mean RMSE: {summary['rmse']['mean']:.2f}")
        
        >>> # Group by model type
        >>> summary_by_model = calculate_summary_statistics(results_df, group_by='model_type')
    """
    if len(results_df) == 0:
        logger.warning("No results to summarize")
        return {}
    
    # Filter out invalid values
    def clean_series(s):
        return s.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Overall statistics
    overall_stats = {}
    for metric in ['mse', 'rmse', 'mae', 'mape']:
        if metric in results_df.columns:
            valid = clean_series(results_df[metric])
            if len(valid) > 0:
                overall_stats[metric] = {
                    'mean': float(valid.mean()),
                    'median': float(valid.median()),
                    'std': float(valid.std()),
                    'min': float(valid.min()),
                    'max': float(valid.max()),
                    'count': len(valid)
                }
    
    summary = {
        'overall': overall_stats,
        'total_wells': len(results_df),
        'total_test_points': int(results_df['n_points'].sum()) if 'n_points' in results_df.columns else None
    }
    
    # Group statistics if requested
    if group_by and group_by in results_df.columns:
        group_stats = {}
        for group_value in results_df[group_by].unique():
            group_df = results_df[results_df[group_by] == group_value]
            group_summary = {}
            
            for metric in ['mse', 'rmse', 'mae', 'mape']:
                if metric in group_df.columns:
                    valid = clean_series(group_df[metric])
                    if len(valid) > 0:
                        group_summary[metric] = {
                            'mean': float(valid.mean()),
                            'median': float(valid.median()),
                            'std': float(valid.std()),
                            'count': len(valid)
                        }
            
            group_stats[str(group_value)] = {
                **group_summary,
                'well_count': len(group_df)
            }
        
        summary['by_' + group_by] = group_stats
    
    return summary


def print_summary_statistics(
    results_df: pd.DataFrame,
    group_by: Optional[str] = None
) -> None:
    """
    Print summary statistics in a readable format.
    
    Args:
        results_df: DataFrame with cross-validation results
        group_by: Optional column name to group by
        
    Example:
        >>> results_df = evaluate_wells_dataset(df, 'well_id', 'date', 'production')
        >>> print_summary_statistics(results_df, group_by='model_type')
    """
    summary = calculate_summary_statistics(results_df, group_by=group_by)
    
    if not summary:
        print("No results to summarize")
        return
    
    print("\n" + "=" * 70)
    print("FORECAST VALIDATION SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal Wells Evaluated: {summary['total_wells']:,}")
    if summary['total_test_points']:
        print(f"Total Test Points: {summary['total_test_points']:,}")
    
    # Overall metrics
    overall = summary.get('overall', {})
    if overall:
        print("\nOverall Metrics:")
        for metric, stats in overall.items():
            metric_name = metric.upper()
            print(f"  {metric_name:>6}: Mean = {stats['mean']:>10.2f}, "
                  f"Median = {stats['median']:>10.2f}, "
                  f"Std = {stats['std']:>10.2f} "
                  f"(n = {stats['count']:,})")
    
    # Group statistics
    group_key = 'by_' + group_by if group_by else None
    if group_key and group_key in summary:
        print(f"\nBy {group_by.title()}:")
        for group_value, group_stats in summary[group_key].items():
            print(f"\n  {group_value}:")
            print(f"    Wells: {group_stats['well_count']:,}")
            
            for metric in ['mse', 'rmse', 'mae', 'mape']:
                if metric in group_stats:
                    stats = group_stats[metric]
                    metric_name = metric.upper()
                    print(f"    {metric_name:>6}: Mean = {stats['mean']:>10.2f}, "
                          f"Median = {stats['median']:>10.2f} (n = {stats['count']:,})")
    
    print("=" * 70 + "\n")

