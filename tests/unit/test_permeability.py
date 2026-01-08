"""
Unit tests for permeability estimation models.
"""

import pytest
import numpy as np
import pandas as pd
from geosuite.petro.permeability import (
    calculate_permeability_kozeny_carman,
    calculate_permeability_timur,
    calculate_permeability_wyllie_rose,
    calculate_permeability_coates_dumanoir,
    calculate_permeability_tixier,
    calculate_permeability_porosity_only,
)


class TestKozenyCarman:
    """Tests for Kozeny-Carman permeability model."""
    
    def test_basic_calculation(self):
        """Test basic permeability calculation."""
        phi = np.array([0.15, 0.20, 0.25])
        k = calculate_permeability_kozeny_carman(phi)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)
        assert np.all(k < 1e6)
    
    def test_with_water_saturation(self):
        """Test with water saturation."""
        phi = np.array([0.20, 0.25])
        sw = np.array([0.30, 0.40])
        k = calculate_permeability_kozeny_carman(phi, sw=sw)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        phi = pd.Series([0.15, 0.20, 0.25])
        k = calculate_permeability_kozeny_carman(phi)
        
        assert len(k) == len(phi)
        assert isinstance(k, np.ndarray)
    
    def test_empty_array(self):
        """Test with empty array."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_permeability_kozeny_carman(np.array([]))
    
    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        phi = np.array([0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_permeability_kozeny_carman(phi, sw=sw)


class TestTimur:
    """Tests for Timur permeability model."""
    
    def test_basic_calculation(self):
        """Test basic permeability calculation."""
        phi = np.array([0.15, 0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        k = calculate_permeability_timur(phi, sw)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)
        assert np.all(k < 1e6)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        phi = pd.Series([0.15, 0.20, 0.25])
        sw = pd.Series([0.30, 0.40, 0.50])
        k = calculate_permeability_timur(phi, sw)
        
        assert len(k) == len(phi)
        assert isinstance(k, np.ndarray)
    
    def test_empty_array(self):
        """Test with empty array."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_permeability_timur(np.array([]), np.array([0.3]))
    
    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        phi = np.array([0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_permeability_timur(phi, sw)
    
    def test_custom_coefficients(self):
        """Test with custom coefficients."""
        phi = np.array([0.20])
        sw = np.array([0.30])
        k = calculate_permeability_timur(phi, sw, coefficient=0.2, porosity_exponent=5.0)
        
        assert k[0] > 0


class TestWyllieRose:
    """Tests for Wyllie-Rose permeability model."""
    
    def test_basic_calculation(self):
        """Test basic permeability calculation."""
        phi = np.array([0.15, 0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        k = calculate_permeability_wyllie_rose(phi, sw)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)
    
    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        phi = np.array([0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_permeability_wyllie_rose(phi, sw)


class TestCoatesDumanoir:
    """Tests for Coates-Dumanoir permeability model."""
    
    def test_basic_calculation(self):
        """Test basic permeability calculation."""
        phi = np.array([0.15, 0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        k = calculate_permeability_coates_dumanoir(phi, sw)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)
    
    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        phi = np.array([0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_permeability_coates_dumanoir(phi, sw)


class TestTixier:
    """Tests for Tixier permeability model."""
    
    def test_basic_calculation(self):
        """Test basic permeability calculation."""
        phi = np.array([0.15, 0.20, 0.25])
        sw = np.array([0.30, 0.40, 0.50])
        k = calculate_permeability_tixier(phi, sw)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)


class TestPorosityOnly:
    """Tests for porosity-only permeability model."""
    
    def test_basic_calculation(self):
        """Test basic permeability calculation."""
        phi = np.array([0.15, 0.20, 0.25])
        k = calculate_permeability_porosity_only(phi)
        
        assert len(k) == len(phi)
        assert np.all(k > 0)
        assert np.all(k < 1e6)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        phi = pd.Series([0.15, 0.20, 0.25])
        k = calculate_permeability_porosity_only(phi)
        
        assert len(k) == len(phi)
        assert isinstance(k, np.ndarray)
    
    def test_empty_array(self):
        """Test with empty array."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_permeability_porosity_only(np.array([]))
    
    def test_custom_coefficients(self):
        """Test with custom coefficients."""
        phi = np.array([0.20])
        k = calculate_permeability_porosity_only(phi, coefficient=200.0, exponent=4.0)
        
        assert k[0] > 0

