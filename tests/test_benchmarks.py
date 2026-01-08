"""
Performance benchmark tests.

These tests track execution time for critical functions to prevent regressions.
"""

import pytest
import numpy as np
from tests.benchmarks import (
    Benchmark,
    benchmark_overburden_stress,
    benchmark_water_saturation,
    benchmark_permeability_timur,
    benchmark_clustering,
    run_all_benchmarks,
)


class TestBenchmarks:
    """Tests for performance benchmarks."""
    
    def test_benchmark_class(self):
        """Test Benchmark utility class."""
        bench = Benchmark("test")
        
        def dummy_func(x):
            return x * 2
        
        elapsed = bench.time_function(dummy_func, 5)
        assert elapsed >= 0
        
        stats = bench.get_stats()
        assert 'mean' in stats
        assert 'n_runs' in stats
    
    def test_benchmark_overburden_stress(self):
        """Test overburden stress benchmark."""
        elapsed = benchmark_overburden_stress(n_samples=1000)
        assert elapsed >= 0
        assert elapsed < 10.0  # Should complete in reasonable time
    
    def test_benchmark_water_saturation(self):
        """Test water saturation benchmark."""
        elapsed = benchmark_water_saturation(n_samples=1000)
        assert elapsed >= 0
        assert elapsed < 10.0
    
    def test_benchmark_permeability_timur(self):
        """Test permeability benchmark."""
        elapsed = benchmark_permeability_timur(n_samples=1000)
        assert elapsed >= 0
        assert elapsed < 10.0
    
    def test_benchmark_clustering(self):
        """Test clustering benchmark."""
        elapsed = benchmark_clustering(n_samples=500, n_features=4)
        assert elapsed >= 0
        assert elapsed < 10.0
    
    def test_run_all_benchmarks(self):
        """Test running all benchmarks."""
        results = run_all_benchmarks(n_runs=2)
        
        assert 'overburden_stress' in results
        assert 'water_saturation' in results
        assert 'permeability_timur' in results
        assert 'clustering' in results
        
        for name, stats in results.items():
            assert 'mean' in stats
            assert stats['n_runs'] == 2

