"""
Unit tests for geomechanics enhancements: stress inversion, fracture orientation, failure criteria.
"""

import pytest
import numpy as np
from geosuite.geomech.stress_inversion import (
    invert_stress_from_breakout,
    invert_stress_from_dif,
    invert_stress_combined,
)
from geosuite.geomech.fracture_orientation import (
    predict_fracture_orientation,
    fracture_orientation_distribution,
    calculate_fracture_aperture,
    calculate_fracture_permeability,
)
from geosuite.geomech.failure_criteria import (
    mohr_coulomb_failure,
    drucker_prager_failure,
    hoek_brown_failure,
    griffith_failure,
    calculate_failure_envelope,
)


class TestStressInversion:
    """Tests for stress inversion functions."""
    
    def test_invert_stress_from_breakout_analytical(self):
        """Test analytical stress inversion from breakout."""
        breakout_width = np.array([30.0, 45.0])
        depth = np.array([1000.0, 2000.0])
        sv = np.array([22.0, 44.0])
        pp = np.array([10.0, 20.0])
        
        result = invert_stress_from_breakout(
            breakout_width, 0.0, depth, sv, pp, method='analytical'
        )
        
        assert 'shmax' in result
        assert 'shmin' in result
        assert 'stress_ratio' in result
        assert len(result['shmax']) == len(breakout_width)
    
    def test_invert_stress_from_dif(self):
        """Test stress inversion from DIF."""
        dif_azimuth = np.array([45.0, 90.0])
        depth = np.array([1000.0, 2000.0])
        sv = np.array([22.0, 44.0])
        pp = np.array([10.0, 20.0])
        
        result = invert_stress_from_dif(dif_azimuth, depth, sv, pp)
        
        assert 'shmax' in result
        assert 'shmin' in result
        assert 'shmin_azimuth' in result
        assert len(result['shmax']) == len(dif_azimuth)
    
    def test_invert_stress_combined(self):
        """Test combined stress inversion."""
        depth = np.array([1000.0, 2000.0])
        sv = np.array([22.0, 44.0])
        pp = np.array([10.0, 20.0])
        
        breakout_data = {
            'width': np.array([30.0, 45.0]),
            'azimuth': np.array([0.0, 0.0])
        }
        
        result = invert_stress_combined(
            depth, sv, pp, breakout_data=breakout_data
        )
        
        assert 'shmax' in result
        assert 'shmin' in result
        assert 'confidence' in result
    
    def test_invert_stress_combined_no_data(self):
        """Test error when no data provided."""
        depth = np.array([1000.0])
        sv = np.array([22.0])
        pp = np.array([10.0])
        
        with pytest.raises(ValueError, match="At least one"):
            invert_stress_combined(depth, sv, pp)


class TestFractureOrientation:
    """Tests for fracture orientation functions."""
    
    def test_predict_fracture_orientation_coulomb(self):
        """Test Coulomb fracture prediction."""
        shmax_azimuth = np.array([0.0, 90.0])
        shmin_azimuth = np.array([90.0, 0.0])
        stress_ratio = np.array([0.8, 1.2])
        
        result = predict_fracture_orientation(
            shmax_azimuth, shmin_azimuth, stress_ratio, method='coulomb'
        )
        
        assert 'strike' in result
        assert 'dip' in result
        assert 'type' in result
        assert len(result['strike']) == len(shmax_azimuth)
    
    def test_predict_fracture_orientation_griffith(self):
        """Test Griffith fracture prediction."""
        shmax_azimuth = np.array([0.0])
        shmin_azimuth = np.array([90.0])
        stress_ratio = np.array([1.0])
        
        result = predict_fracture_orientation(
            shmax_azimuth, shmin_azimuth, stress_ratio, method='griffith'
        )
        
        assert 'strike' in result
        assert result['type'][0] == 'tensile'
    
    def test_fracture_orientation_distribution(self):
        """Test fracture orientation distribution."""
        strikes = fracture_orientation_distribution(mean_strike=45.0, n_samples=100)
        
        assert len(strikes) == 100
        assert np.all((strikes >= 0) & (strikes < 360))
    
    def test_calculate_fracture_aperture(self):
        """Test fracture aperture calculation."""
        normal_stress = np.array([5.0, 10.0, 15.0])
        aperture = calculate_fracture_aperture(normal_stress)
        
        assert len(aperture) == len(normal_stress)
        assert np.all(aperture >= 0)
        assert np.all(aperture <= 0.1)
    
    def test_calculate_fracture_permeability(self):
        """Test fracture permeability calculation."""
        aperture = np.array([0.1, 0.2, 0.3])
        permeability = calculate_fracture_permeability(aperture)
        
        assert len(permeability) == len(aperture)
        assert np.all(permeability > 0)
    
    def test_invalid_method(self):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            predict_fracture_orientation(0.0, 90.0, 1.0, method='invalid')


class TestFailureCriteria:
    """Tests for failure criteria functions."""
    
    def test_mohr_coulomb_failure(self):
        """Test Mohr-Coulomb failure criterion."""
        sigma1 = np.array([50.0, 60.0])
        sigma3 = np.array([10.0, 15.0])
        
        sigma1_fail, safety_factor = mohr_coulomb_failure(sigma1, sigma3)
        
        assert len(sigma1_fail) == len(sigma1)
        assert len(safety_factor) == len(sigma1)
        assert np.all(sigma1_fail > sigma3)
    
    def test_drucker_prager_failure(self):
        """Test Drucker-Prager failure criterion."""
        sigma1 = np.array([50.0])
        sigma2 = np.array([30.0])
        sigma3 = np.array([10.0])
        
        sqrt_J2_fail, safety_factor = drucker_prager_failure(sigma1, sigma2, sigma3)
        
        assert len(sqrt_J2_fail) == 1
        assert safety_factor[0] > 0
    
    def test_hoek_brown_failure(self):
        """Test Hoek-Brown failure criterion."""
        sigma1 = np.array([50.0])
        sigma3 = np.array([10.0])
        
        sigma1_fail, safety_factor = hoek_brown_failure(sigma1, sigma3)
        
        assert len(sigma1_fail) == 1
        assert safety_factor[0] > 0
    
    def test_griffith_failure(self):
        """Test Griffith failure criterion."""
        sigma1 = np.array([20.0, 30.0])
        sigma3 = np.array([5.0, -2.0])
        
        sigma1_fail, safety_factor = griffith_failure(sigma1, sigma3)
        
        assert len(sigma1_fail) == len(sigma1)
        assert len(safety_factor) == len(sigma1)
    
    def test_calculate_failure_envelope(self):
        """Test failure envelope calculation."""
        sigma3_range = np.linspace(0, 50, 10)
        
        sigma3, sigma1_fail = calculate_failure_envelope(
            sigma3_range, criterion='mohr_coulomb'
        )
        
        assert len(sigma3) == len(sigma3_range)
        assert len(sigma1_fail) == len(sigma3_range)
        assert np.all(sigma1_fail > sigma3)
    
    def test_invalid_criterion(self):
        """Test error with invalid criterion."""
        sigma3_range = np.linspace(0, 50, 10)
        
        with pytest.raises(ValueError, match="Unknown criterion"):
            calculate_failure_envelope(sigma3_range, criterion='invalid')

