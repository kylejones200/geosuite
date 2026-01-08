"""
Performance benchmarks to prevent regressions.

These benchmarks track execution time for critical functions.
"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Benchmark:
    """Simple benchmark utility for tracking function performance."""
    
    def __init__(self, name: str):
        self.name = name
        self.times = []
    
    def time_function(self, func: Callable, *args, **kwargs) -> float:
        """
        Time a function execution.
        
        Args:
            func: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Execution time in seconds
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        self.times.append(elapsed)
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics for benchmark times."""
        if len(self.times) == 0:
            return {}
        
        return {
            'mean': np.mean(self.times),
            'std': np.std(self.times),
            'min': np.min(self.times),
            'max': np.max(self.times),
            'n_runs': len(self.times)
        }


def benchmark_overburden_stress(n_samples: int = 10000) -> float:
    """
    Benchmark overburden stress calculation.
    
    Args:
        n_samples: Number of depth samples
        
    Returns:
        Execution time in seconds
    """
    from geosuite.geomech.pressures import calculate_overburden_stress
    
    depth = np.linspace(0, 5000, n_samples)
    rhob = np.full(n_samples, 2.5)
    
    start = time.perf_counter()
    _ = calculate_overburden_stress(depth, rhob)
    elapsed = time.perf_counter() - start
    
    return elapsed


def benchmark_water_saturation(n_samples: int = 10000) -> float:
    """
    Benchmark water saturation calculation.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        Execution time in seconds
    """
    from geosuite.petro.calculations import calculate_water_saturation
    
    phi = np.random.rand(n_samples) * 0.3
    rt = np.random.rand(n_samples) * 100 + 1.0
    
    start = time.perf_counter()
    _ = calculate_water_saturation(phi, rt)
    elapsed = time.perf_counter() - start
    
    return elapsed


def benchmark_permeability_timur(n_samples: int = 10000) -> float:
    """
    Benchmark Timur permeability calculation.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        Execution time in seconds
    """
    from geosuite.petro.permeability import calculate_permeability_timur
    
    phi = np.random.rand(n_samples) * 0.3
    sw = np.random.rand(n_samples) * 0.5 + 0.2
    
    start = time.perf_counter()
    _ = calculate_permeability_timur(phi, sw)
    elapsed = time.perf_counter() - start
    
    return elapsed


def benchmark_clustering(n_samples: int = 1000, n_features: int = 4) -> float:
    """
    Benchmark clustering operation.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Execution time in seconds
    """
    from geosuite.ml.clustering import FaciesClusterer
    
    X = np.random.rand(n_samples, n_features)
    
    start = time.perf_counter()
    clusterer = FaciesClusterer(method='kmeans', n_clusters=5, random_state=42)
    _ = clusterer.fit_predict(X)
    elapsed = time.perf_counter() - start
    
    return elapsed


def run_all_benchmarks(n_runs: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Run all benchmarks multiple times and return statistics.
    
    Args:
        n_runs: Number of runs per benchmark
        
    Returns:
        Dictionary of benchmark statistics
    """
    benchmarks = {
        'overburden_stress': benchmark_overburden_stress,
        'water_saturation': benchmark_water_saturation,
        'permeability_timur': benchmark_permeability_timur,
        'clustering': benchmark_clustering,
    }
    
    results = {}
    
    for name, benchmark_func in benchmarks.items():
        bench = Benchmark(name)
        
        for _ in range(n_runs):
            try:
                bench.time_function(benchmark_func)
            except Exception as e:
                logger.warning(f"Benchmark {name} failed: {e}")
        
        results[name] = bench.get_stats()
    
    return results

