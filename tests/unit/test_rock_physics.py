"""
Unit tests for rock physics transforms.
"""
import pytest
import numpy as np
import pandas as pd

from geosuite.petro.rock_physics import (
    gassmann_fluid_substitution,
    calculate_fluid_bulk_modulus,
    calculate_density_from_velocity,
)


class TestGassmannFluidSubstitution:
    """Tests for Gassmann fluid substitution."""

    def test_gassmann_basic(self):
        """Test basic Gassmann calculation."""
        k_sat_initial = np.array([20.0, 25.0, 30.0])
        k_dry = np.array([15.0, 18.0, 22.0])
        k_mineral = 37.0  # Quartz
        k_fluid_initial = 2.2  # Water
        k_fluid_final = 0.05  # Gas
        phi = np.array([0.2, 0.25, 0.3])
        
        k_sat_final = gassmann_fluid_substitution(
            k_sat_initial, k_dry, k_mineral, k_fluid_initial, k_fluid_final, phi
        )
        
        assert len(k_sat_final) == len(phi)
        # Final bulk modulus should be less than initial (gas is softer)
        assert np.all(k_sat_final < k_sat_initial)

    def test_gassmann_array_inputs(self):
        """Test with array inputs for all parameters."""
        k_sat_initial = np.array([20.0, 25.0])
        k_dry = np.array([15.0, 18.0])
        k_mineral = np.array([37.0, 37.0])
        k_fluid_initial = np.array([2.2, 2.2])
        k_fluid_final = np.array([0.05, 0.05])
        phi = np.array([0.2, 0.25])
        
        k_sat_final = gassmann_fluid_substitution(
            k_sat_initial, k_dry, k_mineral, k_fluid_initial, k_fluid_final, phi
        )
        
        assert len(k_sat_final) == len(phi)

    def test_gassmann_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        k_sat_initial = np.array([20.0, 25.0])
        k_dry = np.array([15.0])
        k_mineral = 37.0
        k_fluid_initial = 2.2
        k_fluid_final = 0.05
        phi = np.array([0.2, 0.25])
        
        with pytest.raises(ValueError, match="same length"):
            gassmann_fluid_substitution(
                k_sat_initial, k_dry, k_mineral, k_fluid_initial, k_fluid_final, phi
            )


class TestFluidBulkModulus:
    """Tests for fluid bulk modulus calculation."""

    def test_fluid_bulk_modulus_water(self):
        """Test with 100% water."""
        sw = np.array([1.0, 1.0, 1.0])
        k_fluid = calculate_fluid_bulk_modulus(sw)
        
        assert len(k_fluid) == len(sw)
        assert np.allclose(k_fluid, 2.2, rtol=0.1)  # Water bulk modulus

    def test_fluid_bulk_modulus_mixed(self):
        """Test with mixed fluid saturations."""
        sw = np.array([0.5, 0.3, 0.7])
        so = np.array([0.3, 0.5, 0.2])
        sg = np.array([0.2, 0.2, 0.1])
        
        k_fluid = calculate_fluid_bulk_modulus(sw, so, sg)
        
        assert len(k_fluid) == len(sw)
        # Mixed fluid should be between water and gas
        assert np.all(k_fluid > 0.05)
        assert np.all(k_fluid < 2.2)

    def test_fluid_bulk_modulus_auto_calculate(self):
        """Test automatic calculation of missing saturations."""
        sw = np.array([0.6, 0.7])
        sg = np.array([0.1, 0.2])
        # so should be calculated as 1 - sw - sg
        
        k_fluid = calculate_fluid_bulk_modulus(sw, sg=sg)
        
        assert len(k_fluid) == len(sw)


class TestDensityFromVelocity:
    """Tests for density estimation from velocity."""

    def test_gardner_method(self):
        """Test Gardner's relation."""
        vp = np.array([2000, 3000, 4000])  # m/s
        rho = calculate_density_from_velocity(vp, method="gardner")
        
        assert len(rho) == len(vp)
        assert np.all(rho > 1.0)
        assert np.all(rho < 3.5)

    def test_nafe_drake_method(self):
        """Test Nafe-Drake relation."""
        vp = np.array([2000, 3000, 4000])
        rho = calculate_density_from_velocity(vp, method="nafe_drake")
        
        assert len(rho) == len(vp)
        assert np.all(rho > 1.0)
        assert np.all(rho <= 3.5)  # Allow 3.5 as it's clipped

    def test_brocher_method(self):
        """Test Brocher's relation."""
        vp = np.array([3000, 4000, 5000])
        rho = calculate_density_from_velocity(vp, method="brocher")
        
        assert len(rho) == len(vp)
        assert np.all(rho > 1.0)
        assert np.all(rho < 3.5)

    def test_invalid_method(self):
        """Test with invalid method."""
        vp = np.array([2000, 3000])
        
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_density_from_velocity(vp, method="invalid")

