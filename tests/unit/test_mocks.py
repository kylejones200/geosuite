"""
Unit tests for mock objects.
"""

import pytest
import numpy as np
import pandas as pd
from tests.mocks import (
    MockWitsmlClient,
    MockMLflowService,
    MockPygeomodelingModel,
    create_mock_las_file,
    create_mock_segy_file,
    patch_optional_dependency,
)


class TestMockWitsmlClient:
    """Tests for MockWitsmlClient."""
    
    def test_connection(self):
        """Test client connection."""
        client = MockWitsmlClient()
        assert not client.connected
        
        result = client.connect("user", "pass")
        assert result is True
        assert client.connected
    
    def test_get_wells(self):
        """Test get wells."""
        client = MockWitsmlClient()
        client.wells = ['Well_001', 'Well_002']
        
        wells = client.get_wells()
        assert len(wells) == 2
    
    def test_get_logs(self):
        """Test get logs."""
        client = MockWitsmlClient()
        client.logs['well_uid_1'] = ['Log_001', 'Log_002']
        
        logs = client.get_logs('well_uid_1')
        assert len(logs) == 2
    
    def test_disconnect(self):
        """Test disconnect."""
        client = MockWitsmlClient()
        client.connect("user", "pass")
        client.disconnect()
        
        assert not client.connected


class TestMockMLflowService:
    """Tests for MockMLflowService."""
    
    def test_create_experiment(self):
        """Test create experiment."""
        service = MockMLflowService()
        exp_id = service.create_experiment("test_exp")
        
        assert exp_id.startswith("exp_")
        assert exp_id in service.experiments
    
    def test_start_run(self):
        """Test start run."""
        service = MockMLflowService()
        exp_id = service.create_experiment("test_exp")
        run_id = service.start_run(exp_id, "test_run")
        
        assert run_id.startswith("run_")
        assert service.current_run == run_id
    
    def test_log_metric(self):
        """Test log metric."""
        service = MockMLflowService()
        exp_id = service.create_experiment("test_exp")
        run_id = service.start_run(exp_id, "test_run")
        
        service.log_metric("accuracy", 0.95)
        assert service.runs[run_id]['metrics']['accuracy'] == 0.95
    
    def test_log_param(self):
        """Test log parameter."""
        service = MockMLflowService()
        exp_id = service.create_experiment("test_exp")
        run_id = service.start_run(exp_id, "test_run")
        
        service.log_param("n_estimators", 100)
        assert service.runs[run_id]['params']['n_estimators'] == 100


class TestMockPygeomodelingModel:
    """Tests for MockPygeomodelingModel."""
    
    def test_fit_and_predict(self):
        """Test fit and predict."""
        model = MockPygeomodelingModel()
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        
        model.fit(X, y)
        assert model.fitted
        
        predictions = model.predict(X)
        assert len(predictions) == 10
    
    def test_predict_with_std(self):
        """Test predict with uncertainty."""
        model = MockPygeomodelingModel()
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        
        model.fit(X, y)
        mean, std = model.predict(X, return_std=True)
        
        assert len(mean) == 10
        assert len(std) == 10
    
    def test_predict_before_fit(self):
        """Test error when predicting before fit."""
        model = MockPygeomodelingModel()
        X = np.random.rand(10, 3)
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)


class TestMockFiles:
    """Tests for mock file objects."""
    
    def test_create_mock_las_file(self):
        """Test mock LAS file."""
        mock_las = create_mock_las_file("test.las")
        
        assert hasattr(mock_las, 'well')
        assert hasattr(mock_las, 'curves')
        assert hasattr(mock_las, 'df')
        
        df = mock_las.df()
        assert isinstance(df, pd.DataFrame)
    
    def test_create_mock_segy_file(self):
        """Test mock SEGY file."""
        mock_segy = create_mock_segy_file("test.sgy")
        
        assert hasattr(mock_segy, 'tracecount')
        assert hasattr(mock_segy, 'samples')
        assert hasattr(mock_segy, 'trace')
        
        trace = mock_segy.trace(0)
        assert len(trace) == 500

