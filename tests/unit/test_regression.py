"""
Unit tests for regression models.
"""
import pytest
import numpy as np
import pandas as pd

from geosuite.ml import PermeabilityPredictor, PorosityPredictor
from geosuite.ml.cross_validation import WellBasedKFold, SpatialCrossValidator


class TestPermeabilityPredictor:
    """Tests for PermeabilityPredictor."""

    def test_fit_and_predict(self):
        """Test fitting and predicting with permeability model."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'GR': np.random.randn(n_samples) * 10 + 50,
            'NPHI': np.random.randn(n_samples) * 0.05 + 0.15,
            'RHOB': np.random.randn(n_samples) * 0.2 + 2.5,
        })
        y = np.random.lognormal(2, 1, n_samples)  # Permeability in mD
        
        predictor = PermeabilityPredictor(model_type='random_forest', test_size=0.2)
        predictor.fit(X, y)
        
        predictions = predictor.predict(X)
        
        assert len(predictions) == len(X)
        assert np.all(predictions >= 0)  # Permeability can't be negative

    def test_different_model_types(self):
        """Test different model types."""
        X = pd.DataFrame({'GR': np.random.randn(50), 'NPHI': np.random.randn(50)})
        y = np.random.lognormal(2, 1, 50)
        
        for model_type in ['random_forest', 'gradient_boosting', 'ridge', 'lasso']:
            predictor = PermeabilityPredictor(model_type=model_type, test_size=0.0)
            predictor.fit(X, y)
            predictions = predictor.predict(X)
            assert len(predictions) == len(X)

    def test_score(self):
        """Test scoring method."""
        X = pd.DataFrame({'GR': np.random.randn(50), 'NPHI': np.random.randn(50)})
        y = np.random.lognormal(2, 1, 50)
        
        predictor = PermeabilityPredictor(test_size=0.0)
        predictor.fit(X, y)
        
        score = predictor.score(X, y)
        assert 0 <= score <= 1  # R² score

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        predictor = PermeabilityPredictor()
        X = pd.DataFrame({'GR': np.random.randn(10), 'NPHI': np.random.randn(10)})
        
        with pytest.raises(ValueError, match="must be fitted"):
            predictor.predict(X)


class TestPorosityPredictor:
    """Tests for PorosityPredictor."""

    def test_fit_and_predict(self):
        """Test fitting and predicting with porosity model."""
        np.random.seed(42)
        X = pd.DataFrame({
            'GR': np.random.randn(50) * 10 + 50,
            'RHOB': np.random.randn(50) * 0.2 + 2.5,
        })
        y = np.random.uniform(0.1, 0.3, 50)  # Porosity
        
        predictor = PorosityPredictor(test_size=0.0)
        predictor.fit(X, y)
        
        predictions = predictor.predict(X)
        
        assert len(predictions) == len(X)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)  # Porosity clipped to [0, 1]


class TestWellBasedKFold:
    """Tests for WellBasedKFold cross-validator."""

    def test_split_by_well(self):
        """Test that splits group by well."""
        df = pd.DataFrame({
            'GR': np.random.randn(100),
            'NPHI': np.random.randn(100),
            'WELL': ['Well_A'] * 30 + ['Well_B'] * 30 + ['Well_C'] * 40,
        })
        
        cv = WellBasedKFold(n_splits=3, well_col='WELL')
        
        splits = list(cv.split(df))
        assert len(splits) == 3
        
        # Check that wells don't appear in both train and test
        for train_idx, test_idx in splits:
            train_wells = set(df.iloc[train_idx]['WELL'].unique())
            test_wells = set(df.iloc[test_idx]['WELL'].unique())
            assert len(train_wells.intersection(test_wells)) == 0

    def test_insufficient_wells(self):
        """Test error when not enough wells."""
        df = pd.DataFrame({
            'GR': np.random.randn(20),
            'WELL': ['Well_A'] * 10 + ['Well_B'] * 10,
        })
        
        cv = WellBasedKFold(n_splits=5, well_col='WELL')
        
        with pytest.raises(ValueError, match="must be >="):
            list(cv.split(df))

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv = WellBasedKFold(n_splits=5)
        assert cv.get_n_splits() == 5


class TestSpatialCrossValidator:
    """Tests for SpatialCrossValidator."""

    def test_split_by_location(self):
        """Test that splits group by spatial location."""
        df = pd.DataFrame({
            'GR': np.random.randn(100),
            'X': np.random.randn(100) * 1000,
            'Y': np.random.randn(100) * 1000,
        })
        
        cv = SpatialCrossValidator(n_splits=5, x_col='X', y_col='Y')
        
        splits = list(cv.split(df))
        assert len(splits) == 5
        
        # Each split should have some data
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_3d_spatial(self):
        """Test 3D spatial cross-validation."""
        df = pd.DataFrame({
            'GR': np.random.randn(100),
            'X': np.random.randn(100) * 1000,
            'Y': np.random.randn(100) * 1000,
            'Z': np.random.randn(100) * 500,
        })
        
        cv = SpatialCrossValidator(n_splits=5, x_col='X', y_col='Y', z_col='Z')
        splits = list(cv.split(df))
        assert len(splits) == 5

