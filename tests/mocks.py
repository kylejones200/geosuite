"""
Mock objects for external dependencies in tests.
"""

from unittest.mock import Mock, MagicMock, patch
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


class MockWitsmlClient:
    """Mock WITSML client for testing."""
    
    def __init__(self, server_url: str = "http://test.witsml.server"):
        self.server_url = server_url
        self.connected = False
        self.wells = []
        self.logs = {}
    
    def connect(self, username: str, password: str) -> bool:
        """Mock connection."""
        self.connected = True
        return True
    
    def get_wells(self) -> list:
        """Mock get wells."""
        return self.wells
    
    def get_logs(self, well_uid: str) -> list:
        """Mock get logs."""
        return self.logs.get(well_uid, [])
    
    def disconnect(self):
        """Mock disconnect."""
        self.connected = False


class MockMLflowService:
    """Mock MLflow service for testing."""
    
    def __init__(self):
        self.experiments = {}
        self.runs = {}
        self.models = {}
        self.current_run = None
    
    def create_experiment(self, name: str) -> str:
        """Mock create experiment."""
        exp_id = f"exp_{len(self.experiments)}"
        self.experiments[exp_id] = {'name': name}
        return exp_id
    
    def start_run(self, experiment_id: str, run_name: str) -> str:
        """Mock start run."""
        run_id = f"run_{len(self.runs)}"
        self.runs[run_id] = {
            'experiment_id': experiment_id,
            'name': run_name,
            'metrics': {},
            'params': {},
        }
        self.current_run = run_id
        return run_id
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Mock log metric."""
        if self.current_run:
            self.runs[self.current_run]['metrics'][key] = value
    
    def log_param(self, key: str, value: Any):
        """Mock log parameter."""
        if self.current_run:
            self.runs[self.current_run]['params'][key] = value
    
    def end_run(self):
        """Mock end run."""
        self.current_run = None


class MockPygeomodelingModel:
    """Mock pygeomodeling model for testing."""
    
    def __init__(self):
        self.fitted = False
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Mock fit."""
        self.X_train = X
        self.y_train = y
        self.fitted = True
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """Mock predict."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        n_samples = len(X)
        mean = np.random.rand(n_samples) * 100
        
        if return_std:
            std = np.random.rand(n_samples) * 10
            return mean, std
        return mean


def create_mock_las_file(file_path: str) -> Mock:
    """
    Create a mock LAS file object.
    
    Args:
        file_path: Path to mock file
        
    Returns:
        Mock LAS file object
    """
    mock_las = Mock()
    mock_las.well = {'WELL': {'value': 'TEST_WELL'}}
    mock_las.curves = ['DEPTH', 'GR', 'RHOB', 'NPHI']
    mock_las.data = np.random.rand(100, 4)
    
    def df():
        return pd.DataFrame(
            mock_las.data,
            columns=mock_las.curves
        )
    
    mock_las.df = df
    return mock_las


def create_mock_segy_file(file_path: str) -> Mock:
    """
    Create a mock SEGY file object.
    
    Args:
        file_path: Path to mock file
        
    Returns:
        Mock SEGY file object
    """
    mock_segy = Mock()
    mock_segy.tracecount = 1000
    mock_segy.samples = 500
    mock_segy.binary = Mock()
    mock_segy.binary = {'sample_interval': 4000}  # microseconds
    
    def read_trace(trace_idx: int):
        return np.random.rand(500)
    
    mock_segy.trace = read_trace
    return mock_segy


def patch_optional_dependency(module_name: str, available: bool = True):
    """
    Context manager to patch optional dependencies for testing.
    
    Args:
        module_name: Name of module to patch
        available: Whether module should be available
        
    Example:
        with patch_optional_dependency('ruptures', available=False):
            # Test code that handles missing ruptures
    """
    if available:
        return patch(module_name)
    else:
        return patch(module_name, side_effect=ImportError(f"No module named '{module_name}'"))

