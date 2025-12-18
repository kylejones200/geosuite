"""
Unit tests for facies classifiers.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from geosuite.ml.classifiers import train_and_predict, FaciesResult


class TestTrainAndPredict:
    """Tests for train_and_predict function."""
    
    def test_basic_training(self, sample_facies_data):
        """Test basic model training and prediction."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM',
            test_size=0.2
        )
        
        assert isinstance(result, FaciesResult)
        assert hasattr(result, 'y_pred')
        assert hasattr(result, 'proba')
        assert hasattr(result, 'classes_')
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'report')
    
    def test_svm_model(self, sample_facies_data):
        """Test SVM model training."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM'
        )
        
        assert result.model_name == "SVM (RBF)"
        assert len(result.y_pred) == len(sample_facies_data)
    
    def test_random_forest_model(self, sample_facies_data):
        """Test Random Forest model training."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='RandomForest'
        )
        
        assert result.model_name == "RandomForest"
        assert len(result.y_pred) == len(sample_facies_data)
    
    def test_with_test_split(self, sample_facies_data):
        """Test training with test set split."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM',
            test_size=0.3
        )
        
        # Should have a classification report
        assert result.report != ""
        assert 'precision' in result.report or 'accuracy' in result.report
    
    def test_probability_output(self, sample_facies_data):
        """Test probability predictions."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM'
        )
        
        assert isinstance(result.proba, pd.DataFrame)
        assert result.proba.shape[0] == len(sample_facies_data)
        
        # Probabilities should sum to 1 (approximately)
        prob_sums = result.proba.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)), decimal=5)
    
    def test_classes_detected(self, sample_facies_data):
        """Test that classes are correctly identified."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM'
        )
        
        unique_facies = sample_facies_data[target_col].unique()
        assert len(result.classes_) > 0
        assert len(result.classes_) <= len(unique_facies)


class TestFaciesResult:
    """Tests for FaciesResult dataclass."""
    
    def test_result_structure(self):
        """Test FaciesResult structure."""
        result = FaciesResult(
            classes_=['A', 'B'],
            y_pred=pd.Series(['A', 'B', 'A']),
            proba=pd.DataFrame([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]),
            model_name='Test Model',
            report='Test report'
        )
        
        assert result.classes_ == ['A', 'B']
        assert len(result.y_pred) == 3
        assert result.proba.shape == (3, 2)
        assert result.model_name == 'Test Model'
        assert result.report == 'Test report'


class TestModelPerformance:
    """Tests for model performance characteristics."""
    
    def test_accuracy_above_random(self, sample_facies_data):
        """Test that model performs better than random."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='RandomForest',
            test_size=0.0  # Use full dataset for simplicity
        )
        
        # Calculate accuracy
        accuracy = accuracy_score(sample_facies_data[target_col], result.y_pred)
        
        # Should be better than random (1 / n_classes)
        n_classes = len(sample_facies_data[target_col].unique())
        random_accuracy = 1.0 / n_classes
        
        assert accuracy > random_accuracy * 1.5  # At least 50% better than random
    
    def test_reproducibility(self, sample_facies_data):
        """Test that results are reproducible with same random state."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        result1 = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM',
            random_state=42
        )
        
        result2 = train_and_predict(
            df=sample_facies_data,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type='SVM',
            random_state=42
        )
        
        pd.testing.assert_series_equal(result1.y_pred, result2.y_pred)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model_type(self, sample_facies_data):
        """Test error handling for invalid model type."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        with pytest.raises(ValueError):
            train_and_predict(
                df=sample_facies_data,
                feature_cols=feature_cols,
                target_col=target_col,
                model_type='InvalidModel'
            )
    
    def test_missing_feature_columns(self, sample_facies_data):
        """Test error handling for missing feature columns."""
        feature_cols = ['NonExistentColumn']
        target_col = 'Facies'
        
        with pytest.raises(KeyError):
            train_and_predict(
                df=sample_facies_data,
                feature_cols=feature_cols,
                target_col=target_col,
                model_type='SVM'
            )
    
    def test_missing_target_column(self, sample_facies_data):
        """Test error handling for missing target column."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'NonExistentTarget'
        
        with pytest.raises(KeyError):
            train_and_predict(
                df=sample_facies_data,
                feature_cols=feature_cols,
                target_col=target_col,
                model_type='SVM'
            )
    
    def test_single_class(self):
        """Test handling of single class data."""
        df = pd.DataFrame({
            'GR': np.random.rand(50),
            'NPHI': np.random.rand(50),
            'Facies': ['A'] * 50  # Only one class
        })
        
        # Should raise an error or handle gracefully
        # (stratify will fail with single class)
        with pytest.raises((ValueError, Exception)):
            train_and_predict(
                df=df,
                feature_cols=['GR', 'NPHI'],
                target_col='Facies',
                model_type='SVM',
                test_size=0.2
            )
    
    def test_small_dataset(self):
        """Test handling of very small dataset."""
        df = pd.DataFrame({
            'GR': [50, 60],
            'NPHI': [0.1, 0.2],
            'Facies': ['A', 'B']
        })
        
        # Should work but with test_size=0
        result = train_and_predict(
            df=df,
            feature_cols=['GR', 'NPHI'],
            target_col='Facies',
            model_type='SVM',
            test_size=0.0
        )
        
        assert len(result.y_pred) == 2


class TestDifferentModelTypes:
    """Test different model configurations."""
    
    def test_all_supported_models(self, sample_facies_data):
        """Test all supported model types."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        model_types = ['SVM', 'RF', 'RandomForest', 'RANDOM_FOREST']
        
        for model_type in model_types:
            result = train_and_predict(
                df=sample_facies_data,
                feature_cols=feature_cols,
                target_col=target_col,
                model_type=model_type,
                test_size=0.0
            )
            
            assert isinstance(result, FaciesResult)
            assert len(result.y_pred) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

