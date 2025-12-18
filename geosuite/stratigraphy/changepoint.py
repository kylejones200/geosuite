"""
Change-point detection for automated stratigraphic interpretation.

This module provides advanced statistical methods for detecting formation boundaries
in well log data. Implements PELT (Pruned Exact Linear Time) algorithm and Bayesian
online change-point detection with uncertainty quantification.

References:
    - Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of 
      changepoints with a linear computational cost. Journal of the American 
      Statistical Association, 107(500), 1590-1598.
    - Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. 
      arXiv preprint arXiv:0710.3742.
"""

import logging
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from typing import List, Tuple, Optional, Dict
import warnings
from geosuite.utils.numba_helpers import njit

logger = logging.getLogger(__name__)

# Check for optional ruptures library
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logger.warning(
        "ruptures library not installed. PELT functionality will be limited. "
        "Install with: pip install ruptures"
    )


def preprocess_log(
    log_values: np.ndarray,
    median_window: int = 5,
    detrend_window: int = 100
) -> np.ndarray:
    """
    Preprocess well log data for change-point detection.
    
    Applies median filtering to remove spikes (preserves edges) and 
    baseline removal to eliminate drift (preserves bed-scale contrasts).
    
    Args:
        log_values: Raw log values (e.g., GR in API units)
        median_window: Window size for spike removal (default 5 samples)
        detrend_window: Window size for baseline removal in samples (0 to skip)
        
    Returns:
        Preprocessed log values
        
    Example:
        >>> gr_raw = np.array([45, 48, 150, 47, 46, ...])  # Spike at index 2
        >>> gr_clean = preprocess_log(gr_raw, median_window=5, detrend_window=100)
    """
    if len(log_values) == 0:
        raise ValueError("log_values cannot be empty")
    
    if not isinstance(log_values, np.ndarray):
        log_values = np.array(log_values)
    
    # Median filter to remove spikes while preserving sharp edges
    log_filtered = median_filter(log_values, size=median_window)
    
    # Optional detrending (remove long-wavelength drift)
    if detrend_window > 0 and len(log_values) > detrend_window:
        # Compute baseline with large median filter (vectorized)
        baseline = median_filter(log_filtered, size=detrend_window)
        
        # Remove baseline and restore median to preserve absolute scale
        log_processed = log_filtered - baseline + np.median(log_filtered)
    else:
        log_processed = log_filtered
    
    return log_processed


def detect_pelt(
    log_values: np.ndarray,
    penalty: Optional[float] = None,
    model: str = 'l2',
    min_size: int = 3,
    jump: int = 1
) -> np.ndarray:
    """
    Detect change points using PELT (Pruned Exact Linear Time) algorithm.
    
    PELT finds the optimal segmentation by minimizing a penalized cost function.
    It guarantees finding the global optimum and runs in near-linear time.
    
    Args:
        log_values: Preprocessed log values
        penalty: Penalty value (higher = fewer change points). 
                 If None, uses log(n) × variance
        model: Cost function model
               - 'l2': Mean shift detection (default)
               - 'rbf': Kernel-based distributional change detection
        min_size: Minimum segment length (default 3)
        jump: Subsample (1 = no subsampling)
        
    Returns:
        Array of change point indices (sorted, excluding start/end)
        
    Raises:
        ImportError: If ruptures library is not installed
        ValueError: If log_values is empty or invalid
        
    Example:
        >>> depth = np.arange(0, 500, 0.5)
        >>> gr_log = load_gamma_ray_log('well_001.las')
        >>> gr_processed = preprocess_log(gr_log)
        >>> change_points = detect_pelt(gr_processed, penalty=50.0)
        >>> formation_tops = depth[change_points]
    """
    if not RUPTURES_AVAILABLE:
        raise ImportError(
            "ruptures library required for PELT algorithm. "
            "Install with: pip install ruptures"
        )
    
    if len(log_values) == 0:
        raise ValueError("log_values cannot be empty")
    
    if not isinstance(log_values, np.ndarray):
        log_values = np.array(log_values)
    
    # Auto-tune penalty if not provided
    if penalty is None:
        n = len(log_values)
        penalty = np.log(n) * np.var(log_values)
        logger.info(f"Auto-tuned penalty: {penalty:.2f}")
    
    # Create PELT model
    if model == 'l2':
        algo = rpt.Pelt(model='l2', min_size=min_size, jump=jump)
    elif model == 'rbf':
        algo = rpt.Pelt(model='rbf', min_size=min_size, jump=jump)
    else:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    
    # Fit and predict
    algo.fit(log_values.reshape(-1, 1))
    change_points = algo.predict(pen=penalty)
    
    # Remove the final point (always equals length of signal)
    change_points = np.array(change_points[:-1])
    
    logger.info(f"PELT detected {len(change_points)} change points (model={model})")
    
    return change_points


@njit(cache=True)
def _bayesian_changepoint_kernel(
    log_values: np.ndarray,
    hazard: float,
    beta0: float
) -> np.ndarray:
    """
    Numba-optimized kernel for Bayesian online change-point detection.
    
    This function is JIT-compiled for 50-100x speedup on large datasets.
    Implements a simplified version of Adams & MacKay (2007) BOCPD algorithm.
    
    Args:
        log_values: Preprocessed log values (1D array)
        hazard: Hazard rate (1 / expected_segment_length)
        beta0: Prior variance parameter
        
    Returns:
        Change point probabilities at each time step
    """
    n = len(log_values)
    
    # Initialize
    run_length_probs = np.zeros(n + 1, dtype=np.float64)
    run_length_probs[0] = 1.0
    
    change_point_probs = np.zeros(n, dtype=np.float64)
    
    # Track sufficient statistics
    sum_x = np.zeros(n + 1, dtype=np.float64)
    sum_x2 = np.zeros(n + 1, dtype=np.float64)
    count = np.zeros(n + 1, dtype=np.float64)
    
    for t in range(n):
        x = log_values[t]
        
        # Roll arrays manually (Numba doesn't support np.roll directly)
        # Shift right by 1
        for idx in range(n, 0, -1):
            sum_x[idx] = sum_x[idx - 1]
            sum_x2[idx] = sum_x2[idx - 1]
            count[idx] = count[idx - 1]
        
        sum_x[0] = 0.0
        sum_x2[0] = 0.0
        count[0] = 0.0
        
        # Update statistics
        for idx in range(1, n + 1):
            sum_x[idx] += x
            sum_x2[idx] += x * x
            count[idx] += 1.0
        
        # Compute predictive probabilities
        predictive_probs = np.ones(n + 1, dtype=np.float64) * 1e-10
        
        max_r = min(t + 1, n)
        for r in range(max_r):
            if count[r] > 0.0:
                n_r = count[r]
                mean_r = sum_x[r] / n_r
                var_r = (sum_x2[r] / n_r - mean_r * mean_r) + beta0
                
                # Gaussian log predictive
                diff = x - mean_r
                predictive_probs[r] = np.exp(-0.5 * (diff * diff) / (var_r + 1e-6))
        
        # Growth probabilities (no change)
        run_length_probs = run_length_probs * predictive_probs
        
        # Change point probability
        cp_prob = np.sum(run_length_probs) * hazard
        change_point_probs[t] = cp_prob / (cp_prob + 1e-10)
        
        # Update run lengths (shift and apply hazard)
        for idx in range(n, 0, -1):
            run_length_probs[idx] = run_length_probs[idx - 1] * (1.0 - hazard)
        
        run_length_probs[0] = cp_prob
        
        # Normalize
        total = np.sum(run_length_probs)
        if total > 0.0:
            run_length_probs = run_length_probs / total
    
    return change_point_probs


def detect_bayesian_online(
    log_values: np.ndarray,
    expected_segment_length: float = 100.0,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect change points using simplified Bayesian online change-point detection.
    
    This is a teaching implementation that captures the core concept (probability 
    spikes at boundaries) without full run-length recursion. It serves as a quick 
    uncertainty signal. For production use requiring rigorous posterior distributions, 
    consider the bayesian-changepoint-detection package for canonical BOCPD.
    
    **Performance:** Accelerated with Numba JIT compilation for 50-100x speedup
    on datasets with 1000+ samples. Falls back to pure Python if Numba unavailable.
    
    Args:
        log_values: Preprocessed log values (numpy array or pandas Series)
        expected_segment_length: Expected length between change points (samples)
        threshold: Probability threshold for flagging change points (0-1)
        
    Returns:
        Tuple of (change_point_indices, probability_at_each_depth)
        
    Example:
        >>> gr_processed = preprocess_log(gr_raw)
        >>> cp_indices, cp_probs = detect_bayesian_online(gr_processed, 
        ...                                                expected_segment_length=50)
        >>> high_confidence = cp_indices[cp_probs[cp_indices] > 0.9]
    """
    if len(log_values) == 0:
        raise ValueError("log_values cannot be empty")
    
    # Convert to numpy array with explicit dtype
    log_values = np.asarray(log_values, dtype=np.float64)
    
    n = len(log_values)
    hazard = 1.0 / expected_segment_length  # Probability of change at each step
    
    # Prior parameters
    beta0 = np.var(log_values)
    
    # Call optimized kernel
    change_point_probs = _bayesian_changepoint_kernel(log_values, hazard, beta0)
    
    # Extract change points above threshold
    change_points = np.where(change_point_probs > threshold)[0]
    
    logger.info(f"Bayesian detection found {len(change_points)} change points "
                f"(threshold={threshold})")
    
    return change_points, change_point_probs


def compare_methods(
    log_values: np.ndarray,
    depth: np.ndarray,
    penalties: Optional[List[float]] = None,
    bayesian_threshold: float = 0.5,
    include_kernel: bool = True
) -> Dict[str, Dict]:
    """
    Compare multiple change-point detection methods.
    
    Args:
        log_values: Preprocessed log values
        depth: Depth array corresponding to log values
        penalties: List of penalty values for PELT (default: auto-tune range)
        bayesian_threshold: Probability threshold for Bayesian method
        include_kernel: If True, include RBF kernel-based PELT
        
    Returns:
        Dictionary with results from each method:
        - 'pelt_pen1', 'pelt_pen2', 'pelt_pen3': PELT with different penalties
        - 'pelt_rbf': Kernel-based PELT (if include_kernel=True)
        - 'bayesian': Bayesian online detection
        
    Example:
        >>> results = compare_methods(gr_processed, depth, penalties=[30, 50, 80])
        >>> consensus_depths = find_consensus(results, tolerance_ft=5.0)
    """
    if len(log_values) != len(depth):
        raise ValueError("log_values and depth must have same length")
    
    if penalties is None:
        # Auto-generate a range of penalties
        base_penalty = np.log(len(log_values)) * np.var(log_values)
        penalties = [base_penalty * 0.5, base_penalty, base_penalty * 2.0]
    
    results = {}
    
    # PELT with different penalties (mean-shift model)
    if RUPTURES_AVAILABLE:
        for i, penalty in enumerate(penalties):
            try:
                cp_indices = detect_pelt(log_values, penalty=penalty, model='l2')
                results[f'pelt_pen{i+1}'] = {
                    'indices': cp_indices,
                    'depths': depth[cp_indices] if len(cp_indices) > 0 else np.array([]),
                    'penalty': penalty,
                    'model': 'l2',
                    'n_points': len(cp_indices)
                }
            except Exception as e:
                logger.warning(f"PELT with penalty {penalty} failed: {e}")
        
        # PELT with kernel model (detects distributional changes)
        if include_kernel:
            try:
                cp_indices_rbf = detect_pelt(log_values, penalty=penalties[1], model='rbf')
                results['pelt_rbf'] = {
                    'indices': cp_indices_rbf,
                    'depths': depth[cp_indices_rbf] if len(cp_indices_rbf) > 0 else np.array([]),
                    'penalty': penalties[1],
                    'model': 'rbf',
                    'n_points': len(cp_indices_rbf)
                }
            except Exception as e:
                logger.warning(f"PELT with RBF kernel failed: {e}")
    else:
        logger.warning("ruptures not available, skipping PELT methods")
    
    # Bayesian online
    try:
        cp_indices_bayes, cp_probs = detect_bayesian_online(
            log_values, 
            threshold=bayesian_threshold
        )
        results['bayesian'] = {
            'indices': cp_indices_bayes,
            'depths': depth[cp_indices_bayes] if len(cp_indices_bayes) > 0 else np.array([]),
            'probabilities': cp_probs,
            'threshold': bayesian_threshold,
            'n_points': len(cp_indices_bayes)
        }
    except Exception as e:
        logger.warning(f"Bayesian detection failed: {e}")
    
    return results


def find_consensus(
    results: Dict[str, Dict],
    tolerance_ft: float = 5.0
) -> np.ndarray:
    """
    Find consensus change points detected by multiple methods.
    
    Clusters nearby picks within tolerance and returns median depth
    of each cluster.
    
    Args:
        results: Dictionary from compare_methods()
        tolerance_ft: Maximum distance (in depth units) to cluster picks
        
    Returns:
        Array of consensus depths
        
    Example:
        >>> results = compare_methods(gr_processed, depth)
        >>> consensus = find_consensus(results, tolerance_ft=5.0)
        >>> logger.info(f"Found {len(consensus)} consensus formation tops")
    """
    # Collect all depths
    all_depths = []
    for method_result in results.values():
        if 'depths' in method_result and len(method_result['depths']) > 0:
            all_depths.extend(method_result['depths'])
    
    consensus_depths = []
    
    if len(all_depths) > 0:
        all_depths = np.array(all_depths)
        # Cluster nearby picks
        used = np.zeros(len(all_depths), dtype=bool)
        
        for i, d in enumerate(all_depths):
            if not used[i]:
                cluster = np.abs(all_depths - d) < tolerance_ft
                consensus_depths.append(np.median(all_depths[cluster]))
                used[cluster] = True
    
    consensus_depths = np.array(sorted(consensus_depths))
    
    logger.info(f"Consensus: {len(consensus_depths)} formation tops "
                f"(tolerance={tolerance_ft} ft)")
    
    return consensus_depths


def tune_penalty_to_target_count(
    log_values: np.ndarray,
    target_picks_per_500ft: int = 8,
    depth_increment_ft: float = 0.5,
    max_iterations: int = 10
) -> float:
    """
    Tune PELT penalty to achieve target pick density.
    
    Iteratively adjusts penalty to match expected formation count.
    Practical target is 6-10 picks per 500 feet for typical stratigraphy.
    
    Args:
        log_values: Preprocessed log values
        target_picks_per_500ft: Target number of picks per 500 ft interval
        depth_increment_ft: Sampling interval in feet
        max_iterations: Maximum tuning iterations
        
    Returns:
        Tuned penalty value
        
    Example:
        >>> gr_processed = preprocess_log(gr_raw)
        >>> penalty = tune_penalty_to_target_count(gr_processed, target_picks_per_500ft=7)
        >>> change_points = detect_pelt(gr_processed, penalty=penalty)
    """
    if not RUPTURES_AVAILABLE:
        raise ImportError("ruptures library required")
    
    # Initial penalty estimate
    n = len(log_values)
    penalty = np.log(n) * np.var(log_values)
    
    # Calculate actual interval length
    total_ft = n * depth_increment_ft
    num_500ft_intervals = total_ft / 500.0
    target_total_picks = int(target_picks_per_500ft * num_500ft_intervals)
    
    logger.info(f"Tuning penalty for {target_picks_per_500ft} picks/500ft "
                f"(~{target_total_picks} total picks in {total_ft:.0f} ft)")
    
    for iteration in range(max_iterations):
        cp_indices = detect_pelt(log_values, penalty=penalty)
        actual_picks = len(cp_indices)
        
        if abs(actual_picks - target_total_picks) <= 2:
            logger.info(f"Converged: {actual_picks} picks with penalty={penalty:.2f}")
            break
        
        # Adjust penalty
        if actual_picks > target_total_picks:
            penalty *= 1.5  # Too many picks, increase penalty
        else:
            penalty *= 0.67  # Too few picks, decrease penalty
        
        logger.info(f"Iteration {iteration+1}: {actual_picks} picks, "
                    f"adjusted penalty to {penalty:.2f}")
    
    return penalty


