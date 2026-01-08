"""
Unit tests for test helpers and utilities.
"""

import pytest
import numpy as np
import pandas as pd
from tests.helpers import (
    assert_well_log_dataframe,
    assert_valid_porosity,
    assert_valid_saturation,
    assert_valid_permeability,
    assert_valid_stress,
    assert_valid_depth,
    assert_array_shape,
    assert_array_lengths_match,
    assert_finite_values,
    assert_close_arrays,
    generate_synthetic_well_log,
    generate_synthetic_facies,
    generate_synthetic_pressure_data,
)


class TestAssertions:
    """Tests for assertion helpers."""
    
    def test_assert_well_log_dataframe_valid(self):
        """Test valid well log DataFrame."""
        df = pd.DataFrame({
            'depth': np.arange(100),
            'GR': np.random.rand(100),
        })
        assert_well_log_dataframe(df)
    
    def test_assert_well_log_dataframe_required_cols(self):
        """Test with required columns."""
        df = pd.DataFrame({
            'depth': np.arange(100),
            'GR': np.random.rand(100),
        })
        assert_well_log_dataframe(df, required_cols=['depth', 'GR'])
    
    def test_assert_well_log_dataframe_missing_cols(self):
        """Test error with missing columns."""
        df = pd.DataFrame({'depth': np.arange(100)})
        with pytest.raises(AssertionError, match="Missing required"):
            assert_well_log_dataframe(df, required_cols=['depth', 'GR'])
    
    def test_assert_valid_porosity(self):
        """Test valid porosity."""
        porosity = np.array([0.1, 0.2, 0.3])
        assert_valid_porosity(porosity)
    
    def test_assert_valid_porosity_invalid(self):
        """Test error with invalid porosity."""
        porosity = np.array([0.1, 1.5, 0.3])
        with pytest.raises(AssertionError):
            assert_valid_porosity(porosity)
    
    def test_assert_valid_saturation(self):
        """Test valid saturation."""
        sw = np.array([0.2, 0.5, 0.8])
        assert_valid_saturation(sw)
    
    def test_assert_valid_permeability(self):
        """Test valid permeability."""
        k = np.array([10.0, 100.0, 1000.0])
        assert_valid_permeability(k)
    
    def test_assert_valid_stress(self):
        """Test valid stress."""
        stress = np.array([10.0, 50.0, 100.0])
        assert_valid_stress(stress)
    
    def test_assert_valid_depth(self):
        """Test valid depth."""
        depth = np.array([1000.0, 2000.0, 3000.0])
        assert_valid_depth(depth)
    
    def test_assert_array_shape(self):
        """Test array shape assertion."""
        arr = np.random.rand(10, 5)
        assert_array_shape(arr, (10, 5))
    
    def test_assert_array_shape_mismatch(self):
        """Test error with shape mismatch."""
        arr = np.random.rand(10, 5)
        with pytest.raises(AssertionError, match="shape"):
            assert_array_shape(arr, (10, 3))
    
    def test_assert_array_lengths_match(self):
        """Test array length matching."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert_array_lengths_match(arr1, arr2)
    
    def test_assert_array_lengths_mismatch(self):
        """Test error with length mismatch."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])
        with pytest.raises(AssertionError, match="don't match"):
            assert_array_lengths_match(arr1, arr2)
    
    def test_assert_finite_values(self):
        """Test finite values assertion."""
        arr = np.array([1.0, 2.0, 3.0])
        assert_finite_values(arr)
    
    def test_assert_finite_values_with_nan(self):
        """Test finite values with NaN allowed."""
        arr = np.array([1.0, np.nan, 3.0])
        assert_finite_values(arr, allow_nan=True)
    
    def test_assert_close_arrays(self):
        """Test close arrays assertion."""
        actual = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.001, 2.001, 3.001])
        assert_close_arrays(actual, expected, rtol=1e-2)


class TestDataGenerators:
    """Tests for data generation helpers."""
    
    def test_generate_synthetic_well_log(self):
        """Test synthetic well log generation."""
        df = generate_synthetic_well_log(n_samples=50)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert 'depth' in df.columns
        assert 'GR' in df.columns
    
    def test_generate_synthetic_facies(self):
        """Test synthetic facies generation."""
        facies = generate_synthetic_facies(n_samples=50, n_facies=3)
        
        assert len(facies) == 50
        assert len(np.unique(facies)) <= 3
    
    def test_generate_synthetic_pressure_data(self):
        """Test synthetic pressure data generation."""
        depth = np.arange(1000, 2000, 10)
        data = generate_synthetic_pressure_data(depth)
        
        assert 'depth' in data
        assert 'ph' in data
        assert 'sv' in data
        assert 'pp' in data
        assert len(data['depth']) == len(depth)

