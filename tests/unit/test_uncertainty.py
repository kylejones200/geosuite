"""
Unit tests for uncertainty quantification utilities.
"""

import pytest
import numpy as np
import pandas as pd
from geosuite.utils.uncertainty import (
    propagate_error,
    confidence_interval,
    monte_carlo_uncertainty,
    uncertainty_porosity_from_density,
    uncertainty_water_saturation,
    uncertainty_permeability,
)


class TestPropagateError:
    """Tests for error propagation."""
    
    def test_simple_function_first_order(self):
        """Test first-order error propagation with simple function."""
        def square(x):
            return x ** 2
        
        x = np.array([2.0, 3.0, 4.0])
        x_error = 0.1
        
        result, uncertainty = propagate_error(square, x, errors=(x_error,), method='first_order')
        
        assert len(result) == len(x)
        assert len(uncertainty) == len(x)
        assert np.all(uncertainty > 0)
    
    def test_simple_function_monte_carlo(self):
        """Test Monte Carlo error propagation."""
        def square(x):
            return x ** 2
        
        x = np.array([2.0, 3.0])
        x_error = 0.1
        
        result, uncertainty = propagate_error(square, x, errors=(x_error,), method='monte_carlo', n_samples=1000)
        
        assert len(result) == len(x)
        assert len(uncertainty) == len(x)
        assert np.all(uncertainty > 0)
    
    def test_multiple_arguments(self):
        """Test error propagation with multiple arguments."""
        def add(x, y):
            return x + y
        
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 3.0])
        
        result, uncertainty = propagate_error(add, x, y, errors=(0.1, 0.1), method='first_order')
        
        assert len(result) == len(x)
        assert len(uncertainty) == len(x)
    
    def test_no_errors(self):
        """Test with no errors provided."""
        def square(x):
            return x ** 2
        
        x = np.array([2.0, 3.0])
        result, uncertainty = propagate_error(square, x, errors=None)
        
        assert np.all(uncertainty == 0)
    
    def test_mismatched_errors(self):
        """Test error when errors don't match arguments."""
        def add(x, y):
            return x + y
        
        x = np.array([1.0])
        y = np.array([2.0])
        
        with pytest.raises(ValueError, match="must match"):
            propagate_error(add, x, y, errors=(0.1,), method='first_order')
    
    def test_invalid_method(self):
        """Test error with invalid method."""
        def square(x):
            return x ** 2
        
        x = np.array([2.0])
        with pytest.raises(ValueError, match="Unknown method"):
            propagate_error(square, x, errors=(0.1,), method='invalid')


class TestConfidenceInterval:
    """Tests for confidence interval calculation."""
    
    def test_basic_confidence_interval(self):
        """Test basic confidence interval calculation."""
        values = np.array([1.0, 2.0, 3.0])
        uncertainty = np.array([0.1, 0.2, 0.3])
        
        lower, upper = confidence_interval(values, uncertainty, confidence=0.95)
        
        assert len(lower) == len(values)
        assert len(upper) == len(values)
        assert np.all(lower < values)
        assert np.all(upper > values)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        values = pd.Series([1.0, 2.0, 3.0])
        uncertainty = pd.Series([0.1, 0.2, 0.3])
        
        lower, upper = confidence_interval(values, uncertainty)
        
        assert len(lower) == len(values)
        assert isinstance(lower, np.ndarray)
    
    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        values = np.array([1.0])
        uncertainty = np.array([0.1])
        
        lower_90, upper_90 = confidence_interval(values, uncertainty, confidence=0.90)
        lower_95, upper_95 = confidence_interval(values, uncertainty, confidence=0.95)
        
        assert (upper_95[0] - lower_95[0]) > (upper_90[0] - lower_90[0])


class TestMonteCarloUncertainty:
    """Tests for Monte Carlo uncertainty analysis."""
    
    def test_normal_distribution(self):
        """Test with normal distribution."""
        def square(x):
            return x ** 2
        
        mean, std = monte_carlo_uncertainty(
            square,
            2.0,
            distributions=('normal',),
            distribution_params=({'scale': 0.1},),
            n_samples=1000
        )
        
        assert isinstance(mean, (float, np.ndarray))
        assert isinstance(std, (float, np.ndarray))
        assert std > 0
    
    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        def square(x):
            return x ** 2
        
        mean, std = monte_carlo_uncertainty(
            square,
            2.0,
            distributions=('uniform',),
            distribution_params=({'low': 1.5, 'high': 2.5},),
            n_samples=1000
        )
        
        assert isinstance(mean, (float, np.ndarray))
        assert std > 0
    
    def test_return_samples(self):
        """Test returning samples."""
        def square(x):
            return x ** 2
        
        mean, std, samples = monte_carlo_uncertainty(
            square,
            2.0,
            distributions=('normal',),
            distribution_params=({'scale': 0.1},),
            n_samples=100,
            return_samples=True
        )
        
        assert len(samples) == 100
        assert isinstance(samples, np.ndarray)
    
    def test_invalid_distribution(self):
        """Test error with invalid distribution."""
        def square(x):
            return x ** 2
        
        with pytest.raises(ValueError, match="Unknown distribution"):
            monte_carlo_uncertainty(
                square,
                2.0,
                distributions=('invalid',),
                distribution_params=({},)
            )


class TestUncertaintyPorosityFromDensity:
    """Tests for porosity uncertainty from density."""
    
    def test_basic_porosity_uncertainty(self):
        """Test basic porosity uncertainty calculation."""
        rhob = np.array([2.3, 2.4, 2.5])
        rhob_error = 0.05
        
        porosity, porosity_error = uncertainty_porosity_from_density(
            rhob, rhob_error, method='first_order'
        )
        
        assert len(porosity) == len(rhob)
        assert len(porosity_error) == len(rhob)
        assert np.all(porosity_error > 0)
    
    def test_monte_carlo_method(self):
        """Test Monte Carlo method."""
        rhob = np.array([2.3, 2.4])
        rhob_error = 0.05
        
        porosity, porosity_error = uncertainty_porosity_from_density(
            rhob, rhob_error, method='monte_carlo', n_samples=1000
        )
        
        assert len(porosity) == len(rhob)
        assert np.all(porosity_error > 0)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        rhob = pd.Series([2.3, 2.4, 2.5])
        rhob_error = 0.05
        
        porosity, porosity_error = uncertainty_porosity_from_density(rhob, rhob_error)
        
        assert len(porosity) == len(rhob)
        assert isinstance(porosity, np.ndarray)


class TestUncertaintyWaterSaturation:
    """Tests for water saturation uncertainty."""
    
    def test_basic_sw_uncertainty(self):
        """Test basic water saturation uncertainty."""
        phi = np.array([0.15, 0.20, 0.25])
        rt = np.array([10.0, 20.0, 30.0])
        phi_error = 0.02
        rt_error = 1.0
        
        sw, sw_error = uncertainty_water_saturation(
            phi, rt, phi_error, rt_error, method='monte_carlo', n_samples=1000
        )
        
        assert len(sw) == len(phi)
        assert len(sw_error) == len(phi)
        assert np.all(sw_error > 0)
    
    def test_with_parameter_uncertainties(self):
        """Test with parameter uncertainties."""
        phi = np.array([0.20])
        rt = np.array([20.0])
        
        sw, sw_error = uncertainty_water_saturation(
            phi, rt, 0.02, 1.0,
            rw=0.05, rw_error=0.01,
            m=2.0, m_error=0.1,
            method='monte_carlo', n_samples=1000
        )
        
        assert len(sw) == 1
        assert sw_error[0] > 0


class TestUncertaintyPermeability:
    """Tests for permeability uncertainty."""
    
    def test_timur_uncertainty(self):
        """Test Timur permeability uncertainty."""
        phi = np.array([0.15, 0.20])
        sw = np.array([0.30, 0.40])
        phi_error = 0.02
        sw_error = 0.05
        
        perm, perm_error = uncertainty_permeability(
            phi, sw, phi_error, sw_error,
            method='timur',
            uncertainty_method='monte_carlo',
            n_samples=1000
        )
        
        assert len(perm) == len(phi)
        assert len(perm_error) == len(phi)
        assert np.all(perm_error > 0)
    
    def test_wyllie_rose_uncertainty(self):
        """Test Wyllie-Rose permeability uncertainty."""
        phi = np.array([0.20, 0.25])
        sw = np.array([0.30, 0.35])
        
        perm, perm_error = uncertainty_permeability(
            phi, sw, 0.02, 0.05,
            method='wyllie_rose',
            uncertainty_method='monte_carlo',
            n_samples=1000
        )
        
        assert len(perm) == len(phi)
        assert np.all(perm_error > 0)
    
    def test_invalid_method(self):
        """Test error with invalid method."""
        phi = np.array([0.20])
        sw = np.array([0.30])
        
        with pytest.raises(ValueError, match="Unknown method"):
            uncertainty_permeability(phi, sw, 0.02, 0.05, method='invalid')

