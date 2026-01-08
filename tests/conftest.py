"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add geosuite_lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test helpers
from .helpers import (
    generate_synthetic_well_log,
    generate_synthetic_facies,
    generate_synthetic_pressure_data,
)


@pytest.fixture
def sample_confusion_matrix():
    """Sample confusion matrix for testing."""
    return np.array([
        [50, 3, 2],
        [2, 45, 5],
        [1, 4, 40]
    ])


@pytest.fixture
def sample_labels():
    """Sample class labels for testing."""
    return ['Sand', 'Shale', 'Siltstone']


@pytest.fixture
def sample_well_log_data():
    """Sample well log data for testing."""
    return pd.DataFrame({
        'DEPTH': np.arange(1000, 1100, 1),
        'GR': np.random.normal(75, 25, 100),
        'NPHI': np.random.normal(0.15, 0.05, 100),
        'RHOB': np.random.normal(2.5, 0.2, 100),
        'PE': np.random.normal(3.0, 0.5, 100)
    })


@pytest.fixture
def sample_facies_data():
    """Sample facies classification data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'DEPTH': np.arange(1000, 1000 + n_samples),
        'GR': np.random.normal(75, 25, n_samples),
        'NPHI': np.random.normal(0.15, 0.05, n_samples),
        'RHOB': np.random.normal(2.5, 0.2, n_samples),
        'PE': np.random.normal(3.0, 0.5, n_samples)
    })
    
    # Add facies labels
    conditions = [
        (data['GR'] < 50) & (data['NPHI'] < 0.1),
        (data['GR'] < 50) & (data['NPHI'] >= 0.1),
        (data['GR'] >= 50) & (data['GR'] < 100),
        data['GR'] >= 100
    ]
    choices = ['Clean_Sand', 'Shaly_Sand', 'Siltstone', 'Shale']
    data['Facies'] = np.select(conditions, choices, default='Unknown')
    
    return data


@pytest.fixture
def flask_app():
    """Create Flask app instance for testing."""
    try:
        import sys
        import os
        # Add webapp to path
        webapp_path = os.path.join(os.path.dirname(__file__), '..', 'webapp')
        if os.path.exists(webapp_path):
            sys.path.insert(0, webapp_path)
            from app import create_app
            
            app = create_app()
            app.config['TESTING'] = True
            return app
        else:
            pytest.skip("Webapp not available")
    except ImportError:
        pytest.skip("Flask app module not available")


@pytest.fixture
def client(flask_app):
    """Create test client."""
    if flask_app is None:
        pytest.skip("Flask app not available")
    return flask_app.test_client()


@pytest.fixture
def adjacent_facies():
    """Sample adjacent facies mapping for testing."""
    return [
        [1],      # Sand adjacent to Shaly_Sand
        [0, 2],   # Shaly_Sand adjacent to Sand and Siltstone
        [1]       # Siltstone adjacent to Shaly_Sand
    ]


@pytest.fixture
def synthetic_well_log():
    """Generate synthetic well log data."""
    return generate_synthetic_well_log(n_samples=100, seed=42)


@pytest.fixture
def synthetic_facies():
    """Generate synthetic facies labels."""
    return generate_synthetic_facies(n_samples=100, seed=42)


@pytest.fixture
def synthetic_pressure_data():
    """Generate synthetic pressure data."""
    depth = np.arange(1000, 2000, 1.0)
    return generate_synthetic_pressure_data(depth)

