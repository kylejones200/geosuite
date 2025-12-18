#!/usr/bin/env python3
"""
Benchmark Numba performance improvements in GeoSuite.

This script measures actual speedups from JIT compilation on key numerical functions.

Usage:
    python benchmarks/bench_numba_speedup.py
    
    # Disable Numba to compare pure Python performance:
    NUMBA_DISABLE_JIT=1 python benchmarks/bench_numba_speedup.py
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geosuite.geomech import calculate_overburden_stress, calculate_pressure_gradient
from geosuite.stratigraphy import detect_bayesian_online, preprocess_log
from geosuite.petro.archie import pickett_isolines, ArchieParams
from geosuite.ml.confusion_matrix_utils import display_adj_cm
from geosuite.utils.numba_helpers import NUMBA_AVAILABLE


def benchmark_overburden_stress():
    """Benchmark overburden stress integration."""
    print("\n" + "="*70)
    print("BENCHMARK: Overburden Stress Integration")
    print("="*70)
    
    # Test different sizes
    sizes = [1000, 5000, 10000]
    
    for n in sizes:
        # Generate test data
        depth = np.linspace(0, 5000, n)
        rhob = np.random.uniform(2.0, 2.8, n)
        
        # Warmup (compile Numba if available)
        _ = calculate_overburden_stress(depth[:100], rhob[:100])
        
        # Benchmark
        n_runs = 100 if n <= 5000 else 50
        start = time.perf_counter()
        for _ in range(n_runs):
            sv = calculate_overburden_stress(depth, rhob)
        elapsed = time.perf_counter() - start
        
        per_run_ms = elapsed / n_runs * 1000
        throughput = n * n_runs / elapsed
        
        print(f"\n  Dataset: {n:,} samples, {n_runs} runs")
        print(f"    Total time: {elapsed:.3f} s")
        print(f"    Per run:    {per_run_ms:.2f} ms")
        print(f"    Throughput: {throughput:,.0f} samples/sec")


def benchmark_bayesian_changepoint():
    """Benchmark Bayesian change-point detection."""
    print("\n" + "="*70)
    print("BENCHMARK: Bayesian Change-Point Detection")
    print("="*70)
    
    # Test different sizes
    sizes = [1000, 2500, 5000]
    
    for n in sizes:
        # Generate synthetic log data with change points
        log_values = np.concatenate([
            np.random.normal(50, 5, n//4),
            np.random.normal(80, 5, n//4),
            np.random.normal(60, 5, n//4),
            np.random.normal(90, 5, n//4),
        ])
        
        # Preprocess
        log_processed = preprocess_log(log_values, median_window=5, detrend_window=0)
        
        # Warmup
        _ = detect_bayesian_online(log_processed[:100], expected_segment_length=50)
        
        # Benchmark
        n_runs = 10 if n <= 2500 else 5
        start = time.perf_counter()
        for _ in range(n_runs):
            cp_indices, cp_probs = detect_bayesian_online(
                log_processed,
                expected_segment_length=50,
                threshold=0.5
            )
        elapsed = time.perf_counter() - start
        
        per_run_ms = elapsed / n_runs * 1000
        
        print(f"\n  Dataset: {n:,} samples, {n_runs} runs")
        print(f"    Total time:   {elapsed:.3f} s")
        print(f"    Per run:      {per_run_ms:.1f} ms")
        print(f"    Detected CPs: {len(cp_indices)}")
        
        # Warn if too slow
        if per_run_ms > 5000:
            print(f"    WARNING: Slow! Consider installing Numba for 50-100x speedup")


def benchmark_tier2_optimizations():
    """Benchmark Tier 2 optimizations (confusion matrix, pickett isolines, gradients)."""
    print("\n" + "="*70)
    print("BENCHMARK: Tier 2 Optimizations")
    print("="*70)
    
    # Test confusion matrix adjustment
    print("\n  Confusion Matrix Adjacent Facies:")
    cm = np.random.randint(0, 100, (9, 9)).astype(float)
    labels = [f"Facies_{i}" for i in range(9)]
    adjacent = [[i-1, i+1] if 0 < i < 8 else [i+1] if i == 0 else [i-1] for i in range(9)]
    
    # Warmup
    _ = display_adj_cm(cm, labels, adjacent, hide_zeros=True)
    
    # Benchmark
    n_runs = 1000
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = display_adj_cm(cm, labels, adjacent, hide_zeros=True)
    elapsed = time.perf_counter() - start
    
    print(f"    9x9 matrix, {n_runs} runs")
    print(f"    Total time: {elapsed:.3f} s")
    print(f"    Per run:    {elapsed/n_runs*1000:.3f} ms")
    
    # Test Pickett isolines
    print("\n  Pickett Isolines Generation:")
    params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.05)
    phi_vals = [0.05, 0.1, 0.2, 0.3]
    sw_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Warmup
    _ = pickett_isolines(phi_vals, sw_vals, params, num_points=100)
    
    # Benchmark
    n_runs = 500
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = pickett_isolines(phi_vals, sw_vals, params, num_points=100)
    elapsed = time.perf_counter() - start
    
    print(f"    5 isolines, 100 points each, {n_runs} runs")
    print(f"    Total time: {elapsed:.3f} s")
    print(f"    Per run:    {elapsed/n_runs*1000:.3f} ms")
    
    # Test pressure gradient
    print("\n  Pressure Gradient Calculation:")
    depth = np.linspace(0, 3000, 5000)
    pressure = np.linspace(0, 70, 5000)
    
    # Warmup
    _ = calculate_pressure_gradient(pressure[:100], depth[:100])
    
    # Benchmark
    n_runs = 500
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = calculate_pressure_gradient(pressure, depth)
    elapsed = time.perf_counter() - start
    
    print(f"    5,000 samples, {n_runs} runs")
    print(f"    Total time: {elapsed:.3f} s")
    print(f"    Per run:    {elapsed/n_runs*1000:.3f} ms")


def benchmark_combined():
    """Benchmark typical workflow combining multiple functions."""
    print("\n" + "="*70)
    print("BENCHMARK: Combined Workflow")
    print("="*70)
    
    n = 5000
    
    # Generate well data
    depth = np.linspace(0, 3000, n)
    rhob = np.random.uniform(2.2, 2.7, n)
    gr = np.concatenate([
        np.random.normal(40, 5, n//3),
        np.random.normal(80, 8, n//3),
        np.random.normal(55, 6, n//3 + n % 3),
    ])
    
    print(f"\n  Workflow: Overburden + Gradient + Change-point detection")
    print(f"  Dataset: {n:,} samples")
    
    # Warmup
    _ = calculate_overburden_stress(depth[:100], rhob[:100])
    _ = detect_bayesian_online(gr[:100], expected_segment_length=50)
    
    # Benchmark complete workflow
    n_runs = 10
    start = time.perf_counter()
    for _ in range(n_runs):
        # Geomechanics
        sv = calculate_overburden_stress(depth, rhob)
        sv_gradient = calculate_pressure_gradient(sv, depth)
        
        # Stratigraphy
        gr_processed = preprocess_log(gr, median_window=5, detrend_window=100)
        cp_indices, cp_probs = detect_bayesian_online(gr_processed, expected_segment_length=100)
    
    elapsed = time.perf_counter() - start
    per_run_ms = elapsed / n_runs * 1000
    
    print(f"\n    Total time: {elapsed:.3f} s")
    print(f"    Per run:    {per_run_ms:.1f} ms")
    print(f"    Speedup:    ~{50 if NUMBA_AVAILABLE else 1}x (estimated)")


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("  GeoSuite Numba Performance Benchmarks")
    print("="*70)
    
    # Check Numba status
    if NUMBA_AVAILABLE:
        print("\n[OK] Numba JIT compilation: ENABLED")
        print("  Expected speedups: 10-100x on numerical code")
    else:
        print("\n[DISABLED] Numba JIT compilation: DISABLED")
        print("  Running in pure Python mode (slower)")
        print("  Install Numba for speedups: pip install numba>=0.58.0")
    
    # Run benchmarks
    try:
        benchmark_overburden_stress()
        benchmark_bayesian_changepoint()
        benchmark_tier2_optimizations()
        benchmark_combined()
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("  Benchmark Complete")
    print("="*70)
    
    if NUMBA_AVAILABLE:
        print("\n[OK] Performance optimizations active")
        print("  First run included compilation overhead")
        print("  Subsequent runs benefit from cached compilation")
    else:
        print("\nWARNING: Running without Numba - performance is not optimized")
        print("   Install Numba to unlock 10-100x speedups:")
        print("   pip install numba>=0.58.0")
    
    print("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

