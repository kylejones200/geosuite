"""
Unit tests for advanced stratigraphy functions.
"""

import pytest
import numpy as np
import pandas as pd
from geosuite.stratigraphy.advanced import (
    ml_segment_timeseries,
    detect_multi_log_boundaries,
    correlate_formations,
)


class TestMLSegmentTimeseries:
    """Tests for ML-based time series segmentation."""
    
    def test_basic_segmentation(self):
        """Test basic ML segmentation."""
        log_data = np.random.rand(100, 3)
        
        result = ml_segment_timeseries(log_data, n_segments=5, method='kmeans')
        
        assert 'labels' in result
        assert 'boundaries' in result
        assert 'n_segments' in result
        assert len(result['labels']) == 100
        assert result['n_segments'] == 5
    
    def test_dataframe_input(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            'GR': np.random.rand(50),
            'RHOB': np.random.rand(50),
            'NPHI': np.random.rand(50),
        })
        
        result = ml_segment_timeseries(df, n_segments=3, features=['GR', 'RHOB'])
        
        assert len(result['labels']) == 50
        assert result['n_segments'] == 3
    
    def test_auto_segments(self):
        """Test automatic segment determination."""
        log_data = np.random.rand(100, 2)
        
        result = ml_segment_timeseries(log_data, n_segments=None, method='kmeans')
        
        assert 'n_segments' in result
        assert result['n_segments'] > 0
    
    def test_pca_kmeans_method(self):
        """Test PCA + KMeans method."""
        log_data = np.random.rand(50, 5)
        
        result = ml_segment_timeseries(log_data, n_segments=4, method='pca_kmeans')
        
        assert len(result['labels']) == 50
    
    def test_hierarchical_method(self):
        """Test hierarchical clustering method."""
        log_data = np.random.rand(50, 2)
        
        result = ml_segment_timeseries(log_data, n_segments=3, method='hierarchical')
        
        assert len(result['labels']) == 50
    
    def test_empty_data(self):
        """Test error with empty data."""
        with pytest.raises(ValueError, match="must not be empty"):
            ml_segment_timeseries(np.array([]).reshape(0, 2))
    
    def test_invalid_method(self):
        """Test error with invalid method."""
        log_data = np.random.rand(50, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            ml_segment_timeseries(log_data, method='invalid')


class TestMultiLogBoundaries:
    """Tests for multi-log boundary detection."""
    
    def test_basic_multi_log_detection(self):
        """Test basic multi-log boundary detection."""
        df = pd.DataFrame({
            'depth': np.arange(0, 500, 0.5),
            'GR': np.random.rand(1000) * 100,
            'RHOB': np.random.rand(1000) * 2 + 2.0,
            'NPHI': np.random.rand(1000) * 0.3,
        })
        
        result = detect_multi_log_boundaries(
            df, log_columns=['GR', 'RHOB'], depth_column='depth'
        )
        
        assert 'consensus_boundaries' in result
        assert 'consensus_depths' in result
        assert 'individual_detections' in result
    
    def test_missing_depth_column(self):
        """Test error when depth column missing."""
        df = pd.DataFrame({
            'GR': np.random.rand(100),
        })
        
        with pytest.raises(ValueError, match="not found"):
            detect_multi_log_boundaries(df, log_columns=['GR'], depth_column='depth')
    
    def test_missing_log_column(self):
        """Test handling of missing log columns."""
        df = pd.DataFrame({
            'depth': np.arange(100),
            'GR': np.random.rand(100),
        })
        
        result = detect_multi_log_boundaries(
            df, log_columns=['GR', 'MISSING'], depth_column='depth'
        )
        
        assert 'consensus_boundaries' in result


class TestCorrelateFormations:
    """Tests for formation correlation."""
    
    def test_basic_correlation(self):
        """Test basic formation correlation."""
        well_data = {
            'Well_001': pd.DataFrame({
                'depth': np.arange(0, 500, 0.5),
                'GR': np.random.rand(1000) * 100,
            }),
            'Well_002': pd.DataFrame({
                'depth': np.arange(0, 500, 0.5),
                'GR': np.random.rand(1000) * 100,
            }),
        }
        
        result = correlate_formations(well_data, method='cross_correlation')
        
        assert 'reference_well' in result
        assert 'correlations' in result
        assert 'Well_002' in result['correlations']
    
    def test_dtw_method(self):
        """Test DTW correlation method."""
        well_data = {
            'Well_001': pd.DataFrame({
                'depth': np.arange(0, 200, 0.5),
                'GR': np.random.rand(400) * 100,
            }),
            'Well_002': pd.DataFrame({
                'depth': np.arange(0, 200, 0.5),
                'GR': np.random.rand(400) * 100,
            }),
        }
        
        result = correlate_formations(well_data, method='dynamic_time_warping')
        
        assert 'correlations' in result
        assert 'Well_002' in result['correlations']
    
    def test_feature_matching_method(self):
        """Test feature matching correlation method."""
        well_data = {
            'Well_001': pd.DataFrame({
                'depth': np.arange(0, 200, 0.5),
                'GR': np.random.rand(400) * 100,
            }),
            'Well_002': pd.DataFrame({
                'depth': np.arange(0, 200, 0.5),
                'GR': np.random.rand(400) * 100,
            }),
        }
        
        result = correlate_formations(well_data, method='feature_matching')
        
        assert 'correlations' in result
    
    def test_insufficient_wells(self):
        """Test error with insufficient wells."""
        well_data = {
            'Well_001': pd.DataFrame({'depth': np.arange(100), 'GR': np.random.rand(100)})
        }
        
        with pytest.raises(ValueError, match="At least two wells"):
            correlate_formations(well_data)
    
    def test_custom_reference_well(self):
        """Test with custom reference well."""
        well_data = {
            'Well_001': pd.DataFrame({
                'depth': np.arange(100),
                'GR': np.random.rand(100),
            }),
            'Well_002': pd.DataFrame({
                'depth': np.arange(100),
                'GR': np.random.rand(100),
            }),
        }
        
        result = correlate_formations(well_data, reference_well='Well_002')
        
        assert result['reference_well'] == 'Well_002'
    
    def test_invalid_reference_well(self):
        """Test error with invalid reference well."""
        well_data = {
            'Well_001': pd.DataFrame({'depth': np.arange(100), 'GR': np.random.rand(100)}),
            'Well_002': pd.DataFrame({'depth': np.arange(100), 'GR': np.random.rand(100)}),
        }
        
        with pytest.raises(ValueError, match="not found"):
            correlate_formations(well_data, reference_well='Invalid')

