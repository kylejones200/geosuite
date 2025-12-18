# GeoSuite Stratigraphy Module

Automated stratigraphic interpretation tools for well log analysis.

## Overview

This module provides advanced statistical methods for detecting formation boundaries and analyzing stratigraphic patterns in well log data. The implementation is based on the refined algorithms from Blog 06 and represents production-quality code suitable for field applications.

## Key Features

- **PELT Algorithm** (Pruned Exact Linear Time) for optimal change-point detection
- **Bayesian Online Detection** with uncertainty quantification
- **Kernel-Based Methods** for distributional change detection
- **Vectorized Preprocessing** for efficient multi-well analysis
- **Auto-Tuning** for penalty parameter calibration
- **Consensus Methods** for combining multiple detection algorithms

## Installation

The `ruptures` library is required for PELT functionality:

```bash
pip install ruptures
```

Basic functionality (Bayesian detection and preprocessing) works without `ruptures`.

## Quick Start

```python
from geosuite.stratigraphy import (
    preprocess_log,
    detect_pelt,
    compare_methods,
    find_consensus
)
import numpy as np

# Load gamma ray log (from LAS file or database)
depth = np.arange(0, 500, 0.5)  # 500 ft at 0.5 ft sampling
gr_log = load_your_gamma_ray_data()

# Preprocess: remove spikes and drift
gr_processed = preprocess_log(gr_log, median_window=5, detrend_window=100)

# Detect change points with PELT
change_points = detect_pelt(gr_processed, penalty=50.0)
formation_tops = depth[change_points]

# Or compare multiple methods
results = compare_methods(gr_processed, depth)
consensus = find_consensus(results, tolerance_ft=5.0)

print(f"Detected {len(consensus)} formation tops")
```

## Functions

### `preprocess_log(log_values, median_window=5, detrend_window=100)`
Preprocess well log data for change-point detection.

- **Median filtering**: Removes noise spikes while preserving sharp edges
- **Baseline removal**: Eliminates tool drift while preserving bed-scale contrasts

**Args:**
- `log_values`: Raw log values (e.g., GR in API units)
- `median_window`: Window size for spike removal (default 5 samples)
- `detrend_window`: Window size for baseline removal (0 to skip)

**Returns:** Preprocessed log values

---

### `detect_pelt(log_values, penalty=None, model='l2')`
Detect change points using PELT algorithm (Killick et al., 2012).

- Guarantees globally optimal segmentation
- Near-linear time complexity
- Supports mean-shift (`model='l2'`) and kernel-based (`model='rbf'`) detection

**Args:**
- `log_values`: Preprocessed log values
- `penalty`: Penalty value (higher = fewer change points). If None, uses log(n) × variance
- `model`: Cost function model ('l2' for mean shift, 'rbf' for distributional changes)

**Returns:** Array of change point indices

---

### `detect_bayesian_online(log_values, expected_segment_length=100.0, threshold=0.5)`
Detect change points with Bayesian online change-point detection (Adams & MacKay, 2007).

- Computes posterior probabilities at each depth
- Provides uncertainty quantification
- Simplified implementation suitable for teaching and quick analysis

**Args:**
- `log_values`: Preprocessed log values
- `expected_segment_length`: Expected length between change points (samples)
- `threshold`: Probability threshold for flagging change points

**Returns:** Tuple of (change_point_indices, probability_at_each_depth)

---

### `compare_methods(log_values, depth, penalties=None, bayesian_threshold=0.5, include_kernel=True)`
Compare multiple change-point detection methods.

Runs PELT with multiple penalties, kernel-based PELT, and Bayesian detection.

**Args:**
- `log_values`: Preprocessed log values
- `depth`: Depth array
- `penalties`: List of penalty values (default: auto-tune range)
- `bayesian_threshold`: Threshold for Bayesian method
- `include_kernel`: If True, include RBF kernel-based PELT

**Returns:** Dictionary with results from each method

---

### `find_consensus(results, tolerance_ft=5.0)`
Find consensus change points detected by multiple methods.

Clusters nearby picks within tolerance and returns median depth of each cluster.

**Args:**
- `results`: Dictionary from `compare_methods()`
- `tolerance_ft`: Maximum distance to cluster picks

**Returns:** Array of consensus depths

---

### `tune_penalty_to_target_count(log_values, target_picks_per_500ft=8, depth_increment_ft=0.5)`
Tune PELT penalty to achieve target pick density.

Practical target is 6-10 picks per 500 feet for typical stratigraphy.

**Args:**
- `log_values`: Preprocessed log values
- `target_picks_per_500ft`: Target number of picks per 500 ft
- `depth_increment_ft`: Sampling interval in feet

**Returns:** Tuned penalty value

## Improvements Over Original Code

This implementation includes several key improvements based on Blog 06 refinements:

### 1. **Vectorized Preprocessing**
- Replaced Python loop with `scipy.ndimage.median_filter()`
- 10-100× faster for multi-well analysis
- Scales efficiently to 47+ wells at 0.5 ft sampling

### 2. **Clearer Parameter Naming**
- `hazard_rate` → `expected_segment_length` (more intuitive)
- Explicit `hazard = 1.0 / expected_segment_length` calculation

### 3. **Production-Ready Error Handling**
- Validates inputs (empty arrays, length mismatches)
- Graceful degradation when `ruptures` unavailable
- Comprehensive logging for debugging

### 4. **Auto-Tuning Utilities**
- `tune_penalty_to_target_count()` for count-based calibration
- Unit-independent tuning rules
- Iterative convergence to target pick density

### 5. **Kernel-Based Detection**
- RBF model added to `detect_pelt()`
- Exposed in `compare_methods()` for distributional changes
- Closes loop between narrative and code delivery

### 6. **Documentation**
- Complete docstrings with examples
- Type hints throughout
- References to Killick et al. (2012) and Adams & MacKay (2007)

## References

- **Killick, R., Fearnhead, P., & Eckley, I. A. (2012).** Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.

- **Adams, R. P., & MacKay, D. J. (2007).** Bayesian online changepoint detection. *arXiv preprint arXiv:0710.3742*.

## Example Use Cases

### 1. Automated Formation Top Picking
```python
# Load 47 Bakken wells
wells = load_well_database()
all_tops = {}

for well_name, gr_log in wells.items():
    gr_processed = preprocess_log(gr_log)
    penalty = tune_penalty_to_target_count(gr_processed, target_picks_per_500ft=7)
    change_points = detect_pelt(gr_processed, penalty=penalty)
    all_tops[well_name] = depth[change_points]

# Consistent picks across all wells in 18 minutes (vs. 120 hours manual)
```

### 2. Uncertainty Quantification
```python
# Get probability curves for QC
cp_indices, cp_probs = detect_bayesian_online(gr_processed)

# High confidence picks (P > 0.9)
high_conf = cp_indices[cp_probs[cp_indices] > 0.9]

# Ambiguous boundaries for review
ambiguous = cp_indices[(cp_probs[cp_indices] > 0.5) & 
                       (cp_probs[cp_indices] < 0.9)]
```

### 3. Multi-Log Integration
```python
# Detect boundaries in GR, resistivity, and density
gr_tops = detect_pelt(preprocess_log(gr_log), penalty=50)
rt_tops = detect_pelt(preprocess_log(rt_log), penalty=40)
rhob_tops = detect_pelt(preprocess_log(rhob_log), penalty=60)

# Find boundaries detected in 2+ logs
from collections import Counter
all_tops = np.concatenate([gr_tops, rt_tops, rhob_tops])
counts = Counter(all_tops)
robust_tops = [top for top, count in counts.items() if count >= 2]
```

## See Also

- **Blog 06**: Change-Point Analysis for Automated Stratigraphic Picks
- **Examples**: `examples/scripts/changepoint_example.py`
- **Related**: `geosuite.ml` for facies classification using detected boundaries


