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
    
    def test_basic_display(self, sample_confusion_matrix, sample_labels):
        """Test basic confusion matrix display."""
        result = display_cm(sample_confusion_matrix, sample_labels)
        
        assert isinstance(result, str)
        assert 'Sand' in result
        assert 'Shale' in result
        assert 'Siltstone' in result
        assert 'Pred' in result
        assert 'True' in result
    
    def test_with_metrics(self, sample_confusion_matrix, sample_labels):
        """Test confusion matrix display with metrics."""
        result = display_cm(
            sample_confusion_matrix, 
            sample_labels, 
            display_metrics=True
        )
        
        assert 'Precision' in result
        assert 'Recall' in result
        assert 'F1' in result
    
    def test_hide_zeros(self, sample_confusion_matrix, sample_labels):
        """Test hiding zero values."""
        # Add a zero to the matrix
        cm = sample_confusion_matrix.copy()
        cm[0, 2] = 0
        
        result = display_cm(cm, sample_labels, hide_zeros=True)
        assert isinstance(result, str)
    
    def test_empty_matrix(self):
        """Test with empty confusion matrix."""
        cm = np.zeros((2, 2))
        labels = ['A', 'B']
        
        result = display_cm(cm, labels)
        assert isinstance(result, str)
    
    def test_single_class(self):
        """Test with single class."""
        cm = np.array([[10]])
        labels = ['Class1']
        
        result = display_cm(cm, labels)
        assert isinstance(result, str)
        assert 'Class1' in result


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
    
    def test_basic_conversion(self, sample_confusion_matrix, sample_labels):
        """Test basic conversion to DataFrame."""
        df = confusion_matrix_to_dataframe(sample_confusion_matrix, sample_labels)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_confusion_matrix.shape
        assert list(df.index) == sample_labels
        assert list(df.columns) == sample_labels
    
    def test_dataframe_values(self, sample_confusion_matrix, sample_labels):
        """Test that DataFrame values match array."""
        df = confusion_matrix_to_dataframe(sample_confusion_matrix, sample_labels)
        
        np.testing.assert_array_equal(df.values, sample_confusion_matrix)
    
    def test_index_names(self, sample_confusion_matrix, sample_labels):
        """Test index and column names."""
        df = confusion_matrix_to_dataframe(sample_confusion_matrix, sample_labels)
        
        assert df.index.name == 'True'
        assert df.columns.name == 'Predicted'


class TestComputeMetricsFromCM:
    """Tests for compute_metrics_from_cm function."""
    
    def test_basic_metrics(self, sample_confusion_matrix, sample_labels):
        """Test basic metrics computation."""
        metrics_df = compute_metrics_from_cm(sample_confusion_matrix, sample_labels)
        
        assert isinstance(metrics_df, pd.DataFrame)
        assert 'Class' in metrics_df.columns
        assert 'Precision' in metrics_df.columns
        assert 'Recall' in metrics_df.columns
        assert 'F1-Score' in metrics_df.columns
        assert 'Support' in metrics_df.columns
    
    def test_metrics_range(self, sample_confusion_matrix, sample_labels):
        """Test that metrics are in valid range [0, 1]."""
        metrics_df = compute_metrics_from_cm(sample_confusion_matrix, sample_labels)
        
        # Exclude the 'Weighted Avg' row and 'Support' column
        metric_cols = ['Precision', 'Recall', 'F1-Score']
        metrics_only = metrics_df[metrics_df['Class'] != 'Weighted Avg'][metric_cols]
        
        assert (metrics_only >= 0).all().all()
        assert (metrics_only <= 1).all().all()
    
    def test_weighted_average(self, sample_confusion_matrix, sample_labels):
        """Test that weighted average is included."""
        metrics_df = compute_metrics_from_cm(sample_confusion_matrix, sample_labels)
        
        assert 'Weighted Avg' in metrics_df['Class'].values
    
    def test_support_values(self, sample_confusion_matrix, sample_labels):
        """Test support values match row sums."""
        metrics_df = compute_metrics_from_cm(sample_confusion_matrix, sample_labels)
        
        # Exclude weighted average
        class_metrics = metrics_df[metrics_df['Class'] != 'Weighted Avg']
        
        expected_support = sample_confusion_matrix.sum(axis=1)
        actual_support = class_metrics['Support'].values
        
        np.testing.assert_array_equal(actual_support, expected_support)
    
    def test_perfect_classifier(self):
        """Test metrics for perfect classifier."""
        cm = np.array([[10, 0], [0, 10]])
        labels = ['A', 'B']
        
        metrics_df = compute_metrics_from_cm(cm, labels)
        class_metrics = metrics_df[metrics_df['Class'] != 'Weighted Avg']
        
        assert (class_metrics['Precision'] == 1.0).all()
        assert (class_metrics['Recall'] == 1.0).all()
        assert (class_metrics['F1-Score'] == 1.0).all()
    
    def test_zero_predictions(self):
        """Test handling of class with zero predictions."""
        cm = np.array([[10, 0], [5, 0]])  # Second class has no predictions
        labels = ['A', 'B']
        
        metrics_df = compute_metrics_from_cm(cm, labels)
        
        # Should handle division by zero gracefully
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
    
    def test_mismatched_dimensions(self):
        """Test handling for mismatched dimensions."""
        cm = np.array([[1, 2], [3, 4]])
        labels = ['A', 'B', 'C']  # More labels than matrix columns
        
        # Should handle gracefully now with updated code
        result = display_cm(cm, labels)
        assert isinstance(result, str)
    
    def test_non_square_matrix(self):
        """Test with non-square matrix."""
        cm = np.array([[1, 2, 3], [4, 5, 6]])
        labels = ['A', 'B', 'C']  # Match number of columns
        
        # Should work but may produce unexpected results
        result = display_cm(cm, labels)
        assert isinstance(result, str)
    
    def test_negative_values(self):
        """Test handling of negative values."""
        cm = np.array([[10, -1], [2, 8]])
        labels = ['A', 'B']
        
        result = display_cm(cm, labels)
        assert isinstance(result, str)
    
    def test_large_matrix(self):
        """Test with large confusion matrix."""
        n_classes = 10
        cm = np.random.randint(0, 100, (n_classes, n_classes))
        labels = [f'Class_{i}' for i in range(n_classes)]
        
        result = display_cm(cm, labels)
        assert isinstance(result, str)
        
        metrics_df = compute_metrics_from_cm(cm, labels)
        assert len(metrics_df) == n_classes + 1  # +1 for weighted avg


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self, sample_confusion_matrix, sample_labels):
        """Test complete workflow: display, convert, compute metrics."""
        # Display
        display_result = display_cm(
            sample_confusion_matrix,
            sample_labels,
            display_metrics=True
        )
        assert isinstance(display_result, str)
        
        # Convert to DataFrame
        df = confusion_matrix_to_dataframe(
            sample_confusion_matrix,
            sample_labels
        )
        assert isinstance(df, pd.DataFrame)
        
        # Compute metrics
        metrics = compute_metrics_from_cm(
            sample_confusion_matrix,
            sample_labels
        )
        assert isinstance(metrics, pd.DataFrame)
        
        # Plot with matplotlib
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_labels
        )
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

