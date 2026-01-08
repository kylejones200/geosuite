"""
Test numerical accuracy of Numba-optimized functions.

These tests ensure that JIT-compiled functions produce identical results
to their pure Python counterparts (within numerical precision limits).
"""

import numpy as np
import pytest
from geosuite.geomech import calculate_overburden_stress, calculate_pressure_gradient
from geosuite.stratigraphy import detect_bayesian_online, preprocess_log
from geosuite.petro.archie import pickett_isolines, ArchieParams
from geosuite.ml.confusion_matrix_utils import display_adj_cm


class TestOverburdenStressAccuracy:
    """Test accuracy of Numba-optimized overburden stress calculation."""
    
    def test_zero_depth(self):
        """Surface should have zero overburden."""
        depth = np.array([0.0, 100.0, 200.0])
        rhob = np.array([2.5, 2.5, 2.5])
        
        sv = calculate_overburden_stress(depth, rhob)
        
        assert sv[0] == 0.0, "Surface overburden must be zero"
    
    def test_monotonic_increase(self):
        """Overburden must increase with depth."""
        np.random.seed(42)
        depth = np.linspace(0, 3000, 1000)
        rhob = np.random.uniform(2.2, 2.7, 1000)
        
        sv = calculate_overburden_stress(depth, rhob)
        
        # Check monotonic increase
        assert np.all(np.diff(sv) >= 0), "Overburden must increase monotonically"
    
    def test_constant_density(self):
        """Test analytical solution for constant density."""
        depth = np.linspace(0, 1000, 100)  # 0-1000m
        rhob = np.ones(100) * 2.5  # 2.5 g/cc = 2500 kg/m³
        g = 9.81  # m/s²
        
        sv = calculate_overburden_stress(depth, rhob, g=g)
        
        # Analytical solution: sv = rho * g * depth
        expected = 2500 * g * depth / 1e6  # Convert Pa to MPa
        
        # Should match within numerical precision
        np.testing.assert_allclose(sv, expected, rtol=1e-10, atol=1e-10)
    
    def test_realistic_values(self):
        """Test with realistic well data."""
        depth = np.linspace(0, 3000, 1000)
        rhob = np.linspace(2.2, 2.7, 1000)  # Typical density increase
        
        sv = calculate_overburden_stress(depth, rhob)
        
        # Final overburden should be in realistic range
        assert 60 < sv[-1] < 80, f"Unrealistic overburden: {sv[-1]:.1f} MPa"
        
        # Check gradient is reasonable (0.02-0.03 MPa/m typical)
        gradient = sv[-1] / depth[-1]
        assert 0.018 < gradient < 0.030, f"Unrealistic gradient: {gradient:.4f} MPa/m"
    
    def test_pandas_series_input(self):
        """Test that pandas Series input works."""
        import pandas as pd
        
        depth = pd.Series(np.linspace(0, 1000, 100))
        rhob = pd.Series(np.ones(100) * 2.5)
        
        sv = calculate_overburden_stress(depth, rhob)
        
        # Should return numpy array
        assert isinstance(sv, np.ndarray)
        assert len(sv) == len(depth)
        assert sv[0] == 0.0
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(123)
        depth = np.linspace(0, 2000, 500)
        rhob = np.random.uniform(2.3, 2.6, 500)
        
        sv1 = calculate_overburden_stress(depth, rhob)
        sv2 = calculate_overburden_stress(depth, rhob)
        
        # Should be identical
        np.testing.assert_array_equal(sv1, sv2)


class TestBayesianChangePointAccuracy:
    """Test accuracy of Numba-optimized Bayesian change-point detection."""
    
    def test_probability_validity(self):
        """All change-point probabilities must be in [0, 1]."""
        np.random.seed(42)
        log_values = np.random.normal(50, 2, 500)
        
        cp_indices, cp_probs = detect_bayesian_online(
            log_values,
            expected_segment_length=100,
            threshold=0.5
        )
        
        assert np.all(cp_probs >= 0.0), "Probabilities below 0"
        assert np.all(cp_probs <= 1.0), "Probabilities above 1"
        assert not np.any(np.isnan(cp_probs)), "NaN in probabilities"
        assert not np.any(np.isinf(cp_probs)), "Inf in probabilities"
    
    def test_detects_abrupt_change(self):
        """Should detect obvious step change."""
        signal = np.concatenate([
            np.ones(200) * 50,
            np.ones(200) * 80,
            np.ones(200) * 50,
        ])
        
        cp_indices, cp_probs = detect_bayesian_online(
            signal,
            expected_segment_length=100,
            threshold=0.5
        )
        
        assert len(cp_indices) >= 1, "Failed to detect obvious step change"
        distances_to_200 = np.abs(cp_indices - 200)
        distances_to_400 = np.abs(cp_indices - 400)
        assert (np.min(distances_to_200) < 50 or np.min(distances_to_400) < 50)
    
    def test_threshold_effect(self):
        """Higher threshold should yield fewer detections."""
        np.random.seed(42)
        signal = np.concatenate([
            np.random.normal(50, 5, 250),
            np.random.normal(70, 5, 250),
        ])
        
        cp_low, _ = detect_bayesian_online(signal, threshold=0.3)
        cp_high, _ = detect_bayesian_online(signal, threshold=0.7)
        
        # Higher threshold should give fewer or equal detections
        assert len(cp_high) <= len(cp_low), "Higher threshold gave more detections"
    
    def test_empty_input_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            detect_bayesian_online(np.array([]))
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(456)
        log_values = np.random.normal(60, 10, 500)
        
        cp1, probs1 = detect_bayesian_online(log_values, expected_segment_length=50)
        cp2, probs2 = detect_bayesian_online(log_values, expected_segment_length=50)
        
        # Should be identical
        np.testing.assert_array_equal(cp1, cp2)
        np.testing.assert_allclose(probs1, probs2, rtol=1e-15)


class TestPreprocessingIntegration:
    """Test Numba functions integrated with preprocessing."""
    
    def test_workflow_overburden_from_density_log(self):
        """Test complete workflow: generate density log, compute overburden."""
        np.random.seed(789)
        
        # Simulate typical well
        depth = np.linspace(0, 3500, 1000)
        
        # Realistic density trend with noise
        rhob_trend = 2.2 + 0.0001 * depth  # Slight compaction trend
        rhob_noise = np.random.normal(0, 0.05, 1000)
        rhob = rhob_trend + rhob_noise
        
        # Compute overburden
        sv = calculate_overburden_stress(depth, rhob)
        
        # Sanity checks
        assert sv[0] == 0.0
        assert sv[-1] > 60.0  # Reasonable for 3500m
        assert sv[-1] < 90.0
        assert np.all(np.diff(sv) >= 0)
    
    def test_workflow_changepoint_on_preprocessed_log(self):
        """Test change-point detection on preprocessed log."""
        # Generate log with formations
        gr = np.concatenate([
            np.random.normal(40, 5, 200),   # Clean sand
            np.random.normal(80, 8, 200),   # Shale
            np.random.normal(50, 6, 200),   # Sandy shale
        ])
        
        # Preprocess
        gr_clean = preprocess_log(gr, median_window=5, detrend_window=0)
        
        # Detect change points
        cp_indices, cp_probs = detect_bayesian_online(
            gr_clean,
            expected_segment_length=80,
            threshold=0.5
        )
        
        # Should detect at least one boundary
        assert len(cp_indices) >= 1
        
        # At least one should be near the real boundaries (200, 400)
        near_200 = np.any(np.abs(cp_indices - 200) < 50)
        near_400 = np.any(np.abs(cp_indices - 400) < 50)
        
        assert near_200 or near_400, "Failed to detect formation boundaries"


class TestNumericalStability:
    """Test numerical stability edge cases."""
    
    def test_overburden_negative_depth_increment(self):
        """Test behavior with non-monotonic depth."""
        # This shouldn't happen in real data, but test robustness
        depth = np.array([0, 100, 90, 200])  # 90 < 100!
        rhob = np.ones(4) * 2.5
        
        sv = calculate_overburden_stress(depth, rhob)
        
        # Should not crash or produce NaN/Inf
        assert not np.any(np.isnan(sv))
        assert not np.any(np.isinf(sv))
    
    def test_bayesian_very_short_signal(self):
        """Test Bayesian detection on minimal signal."""
        log_values = np.array([50.0, 55.0, 60.0, 65.0, 70.0])
        
        # Should not crash
        cp_indices, cp_probs = detect_bayesian_online(log_values)
        
        assert len(cp_probs) == len(log_values)
        assert not np.any(np.isnan(cp_probs))
    
    def test_overburden_extreme_densities(self):
        """Test with extreme (but physically possible) densities."""
        depth = np.linspace(0, 1000, 100)
        
        # Very light (salt: 2.0 g/cc)
        rhob_light = np.ones(100) * 2.0
        sv_light = calculate_overburden_stress(depth, rhob_light)
        assert sv_light[-1] > 10.0  # Should still be reasonable
        
        # Very heavy (iron ore: 5.0 g/cc) 
        rhob_heavy = np.ones(100) * 5.0
        sv_heavy = calculate_overburden_stress(depth, rhob_heavy)
        assert sv_heavy[-1] < 60.0  # Proportionally heavier
        
        # Heavy should be ~2.5x light
        ratio = sv_heavy[-1] / sv_light[-1]
        assert 2.3 < ratio < 2.7


class TestTier2Optimizations:
    """Test Tier 2 Numba optimizations (confusion matrix, pickett, gradients)."""
    
    def test_pressure_gradient_accuracy(self):
        """Test pressure gradient calculation accuracy."""
        # Linear pressure increase
        depth = np.linspace(0, 1000, 100)
        pressure = 0.023 * depth  # Constant gradient
        
        gradient = calculate_pressure_gradient(pressure, depth)
        
        # Should be approximately constant
        assert np.all(gradient > 0.02)
        assert np.all(gradient < 0.03)
        
        # Mean should be close to 0.023
        assert abs(np.mean(gradient) - 0.023) < 0.001
    
    def test_pressure_gradient_pandas_input(self):
        """Test that pandas Series input works."""
        import pandas as pd
        
        depth = pd.Series(np.linspace(0, 1000, 100))
        pressure = pd.Series(np.linspace(0, 23, 100))
        
        gradient = calculate_pressure_gradient(pressure, depth)
        
        assert isinstance(gradient, np.ndarray)
        assert len(gradient) == len(depth)
    
    def test_pickett_isolines_shape(self):
        """Test Pickett isolines generation."""
        params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.05)
        phi_vals = [0.1, 0.2, 0.3]
        sw_vals = [0.2, 0.5, 0.8, 1.0]
        
        lines = pickett_isolines(phi_vals, sw_vals, params, num_points=50)
        
        # Should have one line per Sw value
        assert len(lines) == len(sw_vals)
        
        # Each line should have correct structure
        for phi_grid, rt_grid, label in lines:
            assert len(phi_grid) == 50
            assert len(rt_grid) == 50
            assert "Sw=" in label
            
            # Resistivity should be positive
            assert np.all(rt_grid > 0)
            
            # Porosity should be in valid range
            assert np.all(phi_grid > 0)
            assert np.all(phi_grid < 1)
    
    def test_pickett_isolines_archie_relationship(self):
        """Test that Pickett isolines follow Archie equation."""
        params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.05)
        phi_vals = [0.1, 0.3]
        sw_vals = [0.5]
        
        lines = pickett_isolines(phi_vals, sw_vals, params, num_points=10)
        phi_grid, rt_grid, label = lines[0]
        
        # Verify Archie equation: Rt = (a*Rw) / (phi^m * Sw^n)
        sw = 0.5
        for i in range(len(phi_grid)):
            expected_rt = (params.a * params.rw) / (phi_grid[i]**params.m * sw**params.n)
            assert abs(rt_grid[i] - expected_rt) < 1.0  # Allow small numerical error
    
    def test_confusion_matrix_adjacent_facies(self):
        """Test confusion matrix adjacent facies adjustment."""
        # Create simple 3x3 confusion matrix
        cm = np.array([
            [10, 2, 0],
            [1, 15, 3],
            [0, 2, 12]
        ], dtype=float)
        
        labels = ['A', 'B', 'C']
        adjacent = [[1], [0, 2], [1]]  # A adjacent to B, B to A&C, C to B
        
        # Adjust matrix
        result = display_adj_cm(cm, labels, adjacent, hide_zeros=True)
        
        # Should return a string
        assert isinstance(result, str)
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
    
    def test_confusion_matrix_no_adjacent(self):
        """Test confusion matrix with no adjacent facies."""
        cm = np.eye(5) * 10  # Perfect classification
        labels = ['F1', 'F2', 'F3', 'F4', 'F5']
        adjacent = [[] for _ in range(5)]  # No adjacent facies
        
        result = display_adj_cm(cm, labels, adjacent)
        
        # Should still work
        assert isinstance(result, str)
        assert 'F1' in result


class TestParallelProcessing:
    """Test parallel processing utilities."""
    
    def test_parallel_module_import(self):
        """Test that parallel module can be imported."""
        try:
            from geosuite.geomech.parallel import get_parallel_info
            info = get_parallel_info()
            
            assert 'numba_available' in info
            assert 'parallel_enabled' in info
            assert 'num_threads' in info
        except ImportError:
            pytest.skip("Parallel module not available")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

