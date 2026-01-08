"""
Test helpers and utilities for common assertions and data generation.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
import pytest


def assert_well_log_dataframe(df: pd.DataFrame, required_cols: Optional[List[str]] = None):
    """
    Assert that a DataFrame has the structure of well log data.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
    """
    assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"
    assert not df.empty, "DataFrame must not be empty"
    
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        assert len(missing) == 0, f"Missing required columns: {missing}"


def assert_valid_porosity(porosity: Union[np.ndarray, pd.Series], name: str = "porosity"):
    """
    Assert that porosity values are physically valid.
    
    Args:
        porosity: Porosity array or Series
        name: Name for error messages
    """
    porosity = np.asarray(porosity)
    assert np.all(porosity >= 0), f"{name} must be non-negative"
    assert np.all(porosity <= 1), f"{name} must be <= 1.0 (fraction)"


def assert_valid_saturation(saturation: Union[np.ndarray, pd.Series], name: str = "saturation"):
    """
    Assert that saturation values are physically valid.
    
    Args:
        saturation: Saturation array or Series
        name: Name for error messages
    """
    saturation = np.asarray(saturation)
    assert np.all(saturation >= 0), f"{name} must be non-negative"
    assert np.all(saturation <= 1), f"{name} must be <= 1.0 (fraction)"


def assert_valid_permeability(permeability: Union[np.ndarray, pd.Series], name: str = "permeability"):
    """
    Assert that permeability values are physically valid.
    
    Args:
        permeability: Permeability array or Series
        name: Name for error messages
    """
    permeability = np.asarray(permeability)
    assert np.all(permeability > 0), f"{name} must be positive"
    assert np.all(permeability < 1e6), f"{name} values seem unreasonably high (>1e6 mD)"


def assert_valid_stress(stress: Union[np.ndarray, pd.Series], name: str = "stress"):
    """
    Assert that stress values are physically valid.
    
    Args:
        stress: Stress array or Series (MPa)
        name: Name for error messages
    """
    stress = np.asarray(stress)
    assert np.all(stress > 0), f"{name} must be positive"
    assert np.all(stress < 500), f"{name} values seem unreasonably high (>500 MPa)"


def assert_valid_depth(depth: Union[np.ndarray, pd.Series], name: str = "depth"):
    """
    Assert that depth values are physically valid.
    
    Args:
        depth: Depth array or Series
        name: Name for error messages
    """
    depth = np.asarray(depth)
    assert np.all(depth >= 0), f"{name} must be non-negative"
    assert np.all(depth < 15000), f"{name} values seem unreasonably deep (>15 km)"


def assert_array_shape(array: np.ndarray, expected_shape: tuple, name: str = "array"):
    """
    Assert that an array has the expected shape.
    
    Args:
        array: Array to check
        expected_shape: Expected shape tuple
        name: Name for error messages
    """
    assert array.shape == expected_shape, f"{name} shape {array.shape} != expected {expected_shape}"


def assert_array_lengths_match(*arrays, names: Optional[List[str]] = None):
    """
    Assert that multiple arrays have the same length.
    
    Args:
        *arrays: Arrays to check
        names: Optional names for error messages
    """
    lengths = [len(np.asarray(arr)) for arr in arrays]
    if len(set(lengths)) > 1:
        if names:
            msg = f"Array lengths don't match: {dict(zip(names, lengths))}"
        else:
            msg = f"Array lengths don't match: {lengths}"
        raise AssertionError(msg)


def assert_finite_values(array: Union[np.ndarray, pd.Series], name: str = "array", allow_nan: bool = False):
    """
    Assert that array values are finite.
    
    Args:
        array: Array to check
        name: Name for error messages
        allow_nan: If True, allow NaN values but not inf
    """
    array = np.asarray(array)
    if allow_nan:
        assert np.all(np.isfinite(array) | np.isnan(array)), f"{name} contains non-finite values (inf)"
    else:
        assert np.all(np.isfinite(array)), f"{name} contains non-finite values (nan or inf)"


def assert_close_arrays(
    actual: Union[np.ndarray, pd.Series],
    expected: Union[np.ndarray, pd.Series],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "values"
):
    """
    Assert that two arrays are close (within tolerance).
    
    Args:
        actual: Actual values
        expected: Expected values
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    
    assert actual.shape == expected.shape, f"{name} shapes don't match"
    
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(actual - expected))
        raise AssertionError(
            f"{name} not close: max difference = {max_diff}, "
            f"rtol={rtol}, atol={atol}"
        )


def generate_synthetic_well_log(
    n_samples: int = 100,
    depth_start: float = 1000.0,
    depth_step: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic well log data for testing.
    
    Args:
        n_samples: Number of samples
        depth_start: Starting depth
        depth_step: Depth increment
        seed: Random seed
        
    Returns:
        DataFrame with synthetic well log data
    """
    np.random.seed(seed)
    
    depth = np.arange(depth_start, depth_start + n_samples * depth_step, depth_step)[:n_samples]
    
    df = pd.DataFrame({
        'depth': depth,
        'DEPTH': depth,
        'GR': np.random.normal(75, 25, n_samples),
        'RHOB': np.random.normal(2.5, 0.2, n_samples),
        'NPHI': np.random.normal(0.15, 0.05, n_samples),
        'RT': np.random.lognormal(1.0, 1.0, n_samples),
        'DTC': np.random.normal(200, 50, n_samples),
    })
    
    df['GR'] = np.clip(df['GR'], 0, 200)
    df['RHOB'] = np.clip(df['RHOB'], 1.5, 3.0)
    df['NPHI'] = np.clip(df['NPHI'], 0, 0.5)
    df['RT'] = np.clip(df['RT'], 0.1, 1000)
    
    return df


def generate_synthetic_facies(
    n_samples: int = 100,
    n_facies: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic facies labels for testing.
    
    Args:
        n_samples: Number of samples
        n_facies: Number of facies classes
        seed: Random seed
        
    Returns:
        Array of facies labels
    """
    np.random.seed(seed)
    facies_names = ['Sand', 'Shale', 'Siltstone', 'Carbonate', 'Coal'][:n_facies]
    return np.random.choice(facies_names, size=n_samples)


def generate_synthetic_pressure_data(
    depth: np.ndarray,
    rho_water: float = 1.03,
    g: float = 9.81
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic pressure data for testing.
    
    Args:
        depth: Depth array (m)
        rho_water: Water density (g/cc)
        g: Gravitational acceleration (m/s^2)
        
    Returns:
        Dictionary with pressure data
    """
    depth = np.asarray(depth)
    
    # Hydrostatic pressure
    ph = depth * rho_water * g / 1000.0  # MPa
    
    # Overburden stress (simplified)
    sv = depth * 0.023  # MPa (typical gradient)
    
    # Pore pressure (slightly overpressured)
    pp = ph * 1.1
    
    return {
        'depth': depth,
        'ph': ph,
        'sv': sv,
        'pp': pp
    }

