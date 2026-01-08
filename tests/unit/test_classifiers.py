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
        """Test basic model training and prediction with both model types."""
        feature_cols = ['GR', 'NPHI', 'RHOB', 'PE']
        target_col = 'Facies'
        
        # Check if we have enough samples per class for stratification
        class_counts = sample_facies_data[target_col].value_counts()
        min_class_count = class_counts.min()
        test_size = 0.2 if min_class_count >= 5 else 0.0
        
        for model_type in ['SVM', 'RandomForest']:
            result = train_and_predict(
                df=sample_facies_data,
                feature_cols=feature_cols,
                target_col=target_col,
                model_type=model_type,
                test_size=test_size
            )
            
            assert isinstance(result, FaciesResult)
            assert hasattr(result, 'y_pred')
            assert hasattr(result, 'proba')
            assert hasattr(result, 'classes_')
            assert hasattr(result, 'model_name')
            assert len(result.y_pred) == len(sample_facies_data)
            assert isinstance(result.proba, pd.DataFrame)
            assert result.proba.shape[0] == len(sample_facies_data)
            
            # Probabilities should sum to 1
            prob_sums = result.proba.sum(axis=1)
            np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)), decimal=5)
            
            # Should have classification report when test_size > 0
            if test_size > 0:
                assert result.report != ""
            
            # Classes should be detected
            unique_facies = sample_facies_data[target_col].unique()
            assert len(result.classes_) > 0
            assert len(result.classes_) <= len(unique_facies)


class TestModelPerformance:
    """Tests for model performance characteristics."""
    
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
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Small dataset
        df = pd.DataFrame({
            'GR': [50, 60],
            'NPHI': [0.1, 0.2],
            'Facies': ['A', 'B']
        })
        
        result = train_and_predict(
            df=df,
            feature_cols=['GR', 'NPHI'],
            target_col='Facies',
            model_type='SVM',
            test_size=0.0
        )
        
        assert len(result.y_pred) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

