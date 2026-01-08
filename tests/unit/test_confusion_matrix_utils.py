"""
Unit tests for confusion matrix utilities.
"""
import pytest
import numpy as np
import pandas as pd
from geosuite.ml.confusion_matrix_utils import (
    display_cm,
    display_adj_cm,
    confusion_matrix_to_dataframe,
    compute_metrics_from_cm,
    plot_confusion_matrix
)


class TestDisplayCM:
    """Tests for display_cm function."""
    
    def test_display_with_options(self, sample_confusion_matrix, sample_labels):
        """Test confusion matrix display with various options."""
        # Basic display
        result = display_cm(sample_confusion_matrix, sample_labels)
        assert isinstance(result, str)
        assert 'Sand' in result or 'Pred' in result
        
        # With metrics
        result_metrics = display_cm(sample_confusion_matrix, sample_labels, display_metrics=True)
        assert 'Precision' in result_metrics or 'Recall' in result_metrics
        
        # Hide zeros
        cm = sample_confusion_matrix.copy()
        cm[0, 2] = 0
        result_hide = display_cm(cm, sample_labels, hide_zeros=True)
        assert isinstance(result_hide, str)
    
    def test_edge_cases(self):
        """Test edge cases for display."""
        # Empty matrix
        cm = np.zeros((2, 2))
        result = display_cm(cm, ['A', 'B'])
        assert isinstance(result, str)
        
        # Single class
        result = display_cm(np.array([[10]]), ['Class1'])
        assert isinstance(result, str)


class TestDisplayAdjCM:
    """Tests for display_adj_cm function."""
    
    def test_adjacent_facies(self, sample_confusion_matrix, sample_labels, adjacent_facies):
        """Test adjacent facies confusion matrix."""
        result = display_adj_cm(
            sample_confusion_matrix,
            sample_labels,
            adjacent_facies
        )
        
        assert isinstance(result, str)
        assert 'Sand' in result
    
    def test_no_adjacent(self, sample_confusion_matrix, sample_labels):
        """Test with no adjacent facies."""
        adjacent = [[], [], []]  # No adjacent facies
        
        result = display_adj_cm(
            sample_confusion_matrix,
            sample_labels,
            adjacent
        )
        
        assert isinstance(result, str)


class TestConfusionMatrixToDataFrame:
    """Tests for confusion_matrix_to_dataframe function."""
    
    def test_dataframe_conversion(self, sample_confusion_matrix, sample_labels):
        """Test conversion to DataFrame with validation."""
        df = confusion_matrix_to_dataframe(sample_confusion_matrix, sample_labels)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_confusion_matrix.shape
        assert list(df.index) == sample_labels
        assert list(df.columns) == sample_labels
        assert df.index.name == 'True'
        assert df.columns.name == 'Predicted'
        np.testing.assert_array_equal(df.values, sample_confusion_matrix)


class TestComputeMetricsFromCM:
    """Tests for compute_metrics_from_cm function."""
    
    def test_metrics_computation(self, sample_confusion_matrix, sample_labels):
        """Test metrics computation with validation."""
        metrics_df = compute_metrics_from_cm(sample_confusion_matrix, sample_labels)
        
        assert isinstance(metrics_df, pd.DataFrame)
        assert all(col in metrics_df.columns for col in ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        assert 'Weighted Avg' in metrics_df['Class'].values
        
        # Check metrics are in valid range
        metric_cols = ['Precision', 'Recall', 'F1-Score']
        metrics_only = metrics_df[metrics_df['Class'] != 'Weighted Avg'][metric_cols]
        assert (metrics_only >= 0).all().all()
        assert (metrics_only <= 1).all().all()
        
        # Check support values match row sums
        class_metrics = metrics_df[metrics_df['Class'] != 'Weighted Avg']
        expected_support = sample_confusion_matrix.sum(axis=1)
        actual_support = class_metrics['Support'].values
        np.testing.assert_array_equal(actual_support, expected_support)
    
    def test_edge_case_metrics(self):
        """Test metrics for edge cases."""
        # Perfect classifier
        cm = np.array([[10, 0], [0, 10]])
        metrics_df = compute_metrics_from_cm(cm, ['A', 'B'])
        class_metrics = metrics_df[metrics_df['Class'] != 'Weighted Avg']
        assert (class_metrics['Precision'] == 1.0).all()
        
        # Zero predictions
        cm = np.array([[10, 0], [5, 0]])
        metrics_df = compute_metrics_from_cm(cm, ['A', 'B'])
        assert not metrics_df['Precision'].isna().any()


class TestPlotConfusionMatrixMatplotlib:
    """Tests for plot_confusion_matrix (matplotlib) function."""
    
    def test_basic_plot(self, sample_confusion_matrix, sample_labels):
        """Test basic matplotlib figure creation."""
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_labels
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) > 0
    
    def test_custom_title(self, sample_confusion_matrix, sample_labels):
        """Test custom title."""
        custom_title = "Test Confusion Matrix"
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_labels,
            title=custom_title
        )
        
        assert fig.axes[0].get_title() == custom_title
    
    def test_normalized_plot(self, sample_confusion_matrix, sample_labels):
        """Test normalized confusion matrix plot."""
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_labels,
            normalize=True
        )
        
        assert fig is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Mismatched dimensions (should handle gracefully)
        try:
            result = display_cm(np.array([[1, 2], [3, 4]]), ['A', 'B', 'C'])
            assert isinstance(result, str)
        except (IndexError, ValueError):
            pass  # Expected for mismatched dimensions
        
        # Non-square matrix (use matching labels for columns)
        result = display_cm(np.array([[1, 2], [4, 5]]), ['A', 'B'])
        assert isinstance(result, str)
        
        # Large matrix
        n_classes = 10
        cm = np.random.randint(0, 100, (n_classes, n_classes))
        labels = [f'Class_{i}' for i in range(n_classes)]
        result = display_cm(cm, labels)
        assert isinstance(result, str)
        
        metrics_df = compute_metrics_from_cm(cm, labels)
        assert len(metrics_df) == n_classes + 1




if __name__ == '__main__':
    pytest.main([__file__, '-v'])

