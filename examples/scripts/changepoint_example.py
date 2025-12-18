#!/usr/bin/env python3
"""
Example usage of change-point detection for automated formation top picking.

This script demonstrates how to use the improved change-point detection
methods from the geosuite.stratigraphy module.
"""

import sys
from pathlib import Path

# Add parent directory to path for local development
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


logger.info("Change-Point Detection Example")


# Import GeoSuite stratigraphy module
try:
    from geosuite.stratigraphy import (
        preprocess_log,
        detect_pelt,
        detect_bayesian_online,
        compare_methods,
        find_consensus,
        tune_penalty_to_target_count,
        RUPTURES_AVAILABLE
    )
    logger.info("[OK] Successfully imported geosuite.stratigraphy")
except ImportError as e:
    logger.error(f"[FAIL] Failed to import geosuite.stratigraphy: {e}")
    exit(1)

# Check if ruptures is available
if RUPTURES_AVAILABLE:
    logger.info("[OK] ruptures library is available (PELT will work)")
else:
    logger.warning("[WARNING] ruptures library not installed (PELT unavailable)")
    logger.warning("  Install with: pip install ruptures")

logger.info("")

# Generate synthetic gamma ray log with formation boundaries
logger.info("Generating synthetic gamma ray log...")
np.random.seed(42)

formations = [
    {'name': 'Lodgepole', 'thickness': 80, 'gr_mean': 45, 'gr_std': 8},
    {'name': 'Bakken_Upper', 'thickness': 25, 'gr_mean': 150, 'gr_std': 15},
    {'name': 'Bakken_Middle', 'thickness': 35, 'gr_mean': 60, 'gr_std': 12},
    {'name': 'Bakken_Lower', 'thickness': 30, 'gr_mean': 140, 'gr_std': 18},
    {'name': 'ThreeForks', 'thickness': 80, 'gr_mean': 85, 'gr_std': 12},
]

total_thickness = sum(f['thickness'] for f in formations)
depth = np.arange(0, total_thickness, 0.5)  # 0.5 ft sampling
n_samples = len(depth)

# Build gamma ray log
gr_log = np.zeros(n_samples)
current_idx = 0

for fm in formations:
    n_points = int(fm['thickness'] / 0.5)
    if current_idx + n_points > n_samples:
        n_points = n_samples - current_idx
    
    if n_points > 0:
        gr_log[current_idx:current_idx+n_points] = np.random.normal(
            fm['gr_mean'], fm['gr_std'], n_points
        )
        # Add autocorrelation
        for i in range(current_idx+1, current_idx+n_points):
            gr_log[i] = 0.7 * gr_log[i] + 0.3 * gr_log[i-1]
        current_idx += n_points

# Add noise and spikes
gr_noise = gr_log + np.random.normal(0, 5, n_samples)
spike_indices = np.random.choice(n_samples, 10, replace=False)
gr_noise[spike_indices] += np.random.normal(0, 30, 10)

logger.info(f"  Generated {n_samples} samples over {total_thickness} ft")
logger.info(f"  {len(formations)} formations with distinct GR signatures")


# Step 1: Preprocess the log
logger.info("Step 1: Preprocessing log...")
gr_processed = preprocess_log(gr_noise, median_window=5, detrend_window=100)
logger.info("  Applied median filter (removes spikes, preserves edges)")
logger.info("  Applied baseline removal (removes drift, preserves contrasts)")


# Step 2: Detect change points with PELT
if RUPTURES_AVAILABLE:
    logger.info("Step 2: Detecting change points with PELT...")
    
    # Auto-tuned penalty
    penalty_auto = np.log(n_samples) * np.var(gr_processed)
    cp_auto = detect_pelt(gr_processed, penalty=penalty_auto)
    logger.info(f"  Auto-tuned penalty: {penalty_auto:.2f} → {len(cp_auto)} picks")
    
    # Tune to target density
    logger.info("  Tuning penalty for target pick density (8 picks per 500 ft)...")
    penalty_tuned = tune_penalty_to_target_count(
        gr_processed, 
        target_picks_per_500ft=8,
        depth_increment_ft=0.5
    )
    cp_tuned = detect_pelt(gr_processed, penalty=penalty_tuned)
    logger.info(f"  Tuned penalty: {penalty_tuned:.2f} → {len(cp_tuned)} picks")
    
    # Try kernel-based detection
    logger.info("  Running kernel-based PELT (RBF)...")
    cp_rbf = detect_pelt(gr_processed, penalty=penalty_tuned, model='rbf')
    logger.info(f"  RBF model → {len(cp_rbf)} picks")
    
else:
    logger.info("Step 2: Skipping PELT (ruptures not available)")
    

# Step 3: Bayesian detection
logger.info("Step 3: Bayesian online change-point detection...")
cp_bayes, cp_probs = detect_bayesian_online(
    gr_processed, 
    expected_segment_length=50.0,
    threshold=0.5
)
logger.info(f"  Found {len(cp_bayes)} change points")
logger.info(f"  Max probability: {cp_probs.max():.3f}")


# Step 4: Compare methods
if RUPTURES_AVAILABLE:
    logger.info("Step 4: Comparing multiple methods...")
    results = compare_methods(
        gr_processed,
        depth,
        penalties=[penalty_tuned * 0.5, penalty_tuned, penalty_tuned * 2.0],
        bayesian_threshold=0.5,
        include_kernel=True
    )
    logger.info(f"  Ran {len(results)} detection methods")
    
    
    # Step 5: Find consensus
    logger.info("Step 5: Finding consensus picks...")
    consensus = find_consensus(results, tolerance_ft=5.0)
    logger.info(f"  Consensus: {len(consensus)} formation tops")
    
    logger.info("Consensus formation tops (ft):")
    for i, top_depth in enumerate(consensus):
        logger.info(f"  {i+1:2d}. {top_depth:7.1f} ft")
    
else:
    logger.info("Step 4: Skipping method comparison (ruptures not available)")
    

# Summary

logger.info("Summary")

logger.info(f"Input: {n_samples} samples, {total_thickness} ft interval")
logger.info(f"True formations: {len(formations)}")
if RUPTURES_AVAILABLE:
    logger.info(f"Detected (PELT tuned): {len(cp_tuned)} change points")
    logger.info(f"Detected (Bayesian): {len(cp_bayes)} change points")
    logger.info(f"Consensus picks: {len(consensus)} formation tops")
else:
    logger.info(f"Detected (Bayesian only): {len(cp_bayes)} change points")

logger.info("Example complete!")
logger.info("For production use, load real LAS files with:")
logger.info("  from geosuite.io import read_las")
logger.info("  df = read_las('well_001.las')")
logger.info("  gr_processed = preprocess_log(df['GR'].values)")
