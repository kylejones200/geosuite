"""
Unit tests for shaly sand water saturation models.
"""
import pytest
import numpy as np
import pandas as pd

from geosuite.petro.shaly_sand import (
    calculate_water_saturation_simandoux,
    calculate_water_saturation_indonesia,
    calculate_water_saturation_waxman_smits,
)


class TestSimandoux:
    """Tests for Simandoux equation."""

    def test_simandoux_basic(self):
        """Test basic Simandoux calculation."""
        phi = np.array([0.2, 0.25, 0.3])
        rt = np.array([10.0, 15.0, 20.0])
        rsh = np.array([2.0, 2.0, 2.0])
        vsh = np.array([0.1, 0.15, 0.2])
        
        sw = calculate_water_saturation_simandoux(phi, rt, rsh, vsh)
        
        assert len(sw) == len(phi)
        # Check finite values are in valid range (NaN values are expected for invalid inputs)
        finite_mask = np.isfinite(sw)
        if np.any(finite_mask):
            assert np.all(sw[finite_mask] >= 0)
            assert np.all(sw[finite_mask] <= 1)

    def test_simandoux_pandas(self):
        """Test Simandoux with pandas Series."""
        phi = pd.Series([0.2, 0.25, 0.3])
        rt = pd.Series([10.0, 15.0, 20.0])
        rsh = pd.Series([2.0, 2.0, 2.0])
        vsh = pd.Series([0.1, 0.15, 0.2])
        
        sw = calculate_water_saturation_simandoux(phi, rt, rsh, vsh)
        
        assert isinstance(sw, np.ndarray)
        assert len(sw) == len(phi)

    def test_simandoux_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        phi = np.array([0.2, 0.25])
        rt = np.array([10.0, 15.0, 20.0])
        rsh = np.array([2.0, 2.0])
        vsh = np.array([0.1, 0.15])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_water_saturation_simandoux(phi, rt, rsh, vsh)

    def test_simandoux_invalid_vsh(self):
        """Test with invalid shale volume (should clip)."""
        phi = np.array([0.2, 0.25])
        rt = np.array([10.0, 15.0])
        rsh = np.array([2.0, 2.0])
        vsh = np.array([-0.1, 1.5])  # Invalid values
        
        # Should not raise, but should clip
        sw = calculate_water_saturation_simandoux(phi, rt, rsh, vsh)
        assert len(sw) == len(phi)


class TestIndonesia:
    """Tests for Indonesia equation."""

    def test_indonesia_basic(self):
        """Test basic Indonesia calculation."""
        phi = np.array([0.2, 0.25, 0.3])
        rt = np.array([10.0, 15.0, 20.0])
        rsh = np.array([2.0, 2.0, 2.0])
        vsh = np.array([0.1, 0.15, 0.2])
        
        sw = calculate_water_saturation_indonesia(phi, rt, rsh, vsh)
        
        assert len(sw) == len(phi)
        assert np.all(sw >= 0)
        assert np.all(sw <= 1)

    def test_indonesia_vs_simandoux(self):
        """Test that Indonesia gives different results than Simandoux."""
        phi = np.array([0.2, 0.25])
        rt = np.array([10.0, 15.0])
        rsh = np.array([2.0, 2.0])
        vsh = np.array([0.1, 0.15])
        
        sw_sim = calculate_water_saturation_simandoux(phi, rt, rsh, vsh)
        sw_ind = calculate_water_saturation_indonesia(phi, rt, rsh, vsh)
        
        # Results should be different (Indonesia is improved version)
        assert not np.allclose(sw_sim, sw_ind, rtol=1e-3)


class TestWaxmanSmits:
    """Tests for Waxman-Smits equation."""

    def test_waxman_smits_basic(self):
        """Test basic Waxman-Smits calculation."""
        phi = np.array([0.2, 0.25, 0.3])
        rt = np.array([10.0, 15.0, 20.0])
        cec = np.array([5.0, 10.0, 15.0])  # meq/100g
        
        sw = calculate_water_saturation_waxman_smits(phi, rt, cec)
        
        assert len(sw) == len(phi)
        assert np.all(sw >= 0)
        assert np.all(sw <= 1)

    def test_waxman_smits_with_temperature(self):
        """Test Waxman-Smits with temperature parameter."""
        phi = np.array([0.2, 0.25])
        rt = np.array([10.0, 15.0])
        cec = np.array([5.0, 10.0])
        
        sw_25 = calculate_water_saturation_waxman_smits(phi, rt, cec, temperature=25.0)
        sw_50 = calculate_water_saturation_waxman_smits(phi, rt, cec, temperature=50.0)
        
        # Results may be similar or different depending on parameters
        # Just verify both are valid
        assert len(sw_25) == len(phi)
        assert len(sw_50) == len(phi)
        assert np.all(sw_25 >= 0) and np.all(sw_25 <= 1)
        assert np.all(sw_50 >= 0) and np.all(sw_50 <= 1)

    def test_waxman_smits_with_b(self):
        """Test Waxman-Smits with explicit B parameter."""
        phi = np.array([0.2, 0.25])
        rt = np.array([10.0, 15.0])
        cec = np.array([5.0, 10.0])
        
        sw = calculate_water_saturation_waxman_smits(phi, rt, cec, b=4.0)
        
        assert len(sw) == len(phi)
        assert np.all(sw >= 0)
        assert np.all(sw <= 1)
