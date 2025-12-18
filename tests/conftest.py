"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add geosuite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
    n_samples = 200  # Increase sample size to ensure enough per class
    
    data = pd.DataFrame({
        'DEPTH': np.arange(1000, 1000 + n_samples),
        'GR': np.random.normal(75, 30, n_samples),  # Increase variance
        'NPHI': np.random.normal(0.15, 0.08, n_samples),  # Increase variance
        'RHOB': np.random.normal(2.5, 0.2, n_samples),
        'PE': np.random.normal(3.0, 0.5, n_samples)
    })
    
    # Add facies labels with broader conditions to ensure multiple samples per class
    conditions = [
        (data['GR'] < 55),  # More liberal condition for Clean_Sand
        (data['GR'] >= 55) & (data['GR'] < 75),
        (data['GR'] >= 75) & (data['GR'] < 95),
        data['GR'] >= 95
    ]
    choices = ['Clean_Sand', 'Shaly_Sand', 'Siltstone', 'Shale']
    data['Facies'] = np.select(conditions, choices, default='Siltstone')
    
    # Ensure we have at least 10 samples per class for proper stratification
    for facies in choices:
        if (data['Facies'] == facies).sum() < 10:
            # Add more samples for this facies
            indices = data.sample(10, random_state=42).index
            data.loc[indices, 'Facies'] = facies
    
    return data


@pytest.fixture
def flask_app():
    """Create Flask app instance for testing."""
    # Add webapp directory to path
    webapp_path = os.path.join(os.path.dirname(__file__), '..', 'webapp')
    if webapp_path not in sys.path:
        sys.path.insert(0, webapp_path)
    
    try:
        from app import create_app
        
        app = create_app()
        app.config['TESTING'] = True
        app.config['TRAP_HTTP_EXCEPTIONS'] = False  # Don't raise exceptions
        
        # Register error handler for template errors
        @app.errorhandler(500)
        def handle_500(error):
            return "Internal Server Error", 500
            
        @app.errorhandler(404)
        def handle_404(error):
            return "Not Found", 404
        
        return app
    except ImportError as e:
        pytest.skip(f"Flask app not available for testing: {e}")


@pytest.fixture
def client(flask_app):
    """Create test client."""
    return flask_app.test_client()


@pytest.fixture
def adjacent_facies():
    """Sample adjacent facies mapping for testing."""
    return [
        [1],      # Sand adjacent to Shaly_Sand
        [0, 2],   # Shaly_Sand adjacent to Sand and Siltstone
        [1]       # Siltstone adjacent to Shaly_Sand
    ]

