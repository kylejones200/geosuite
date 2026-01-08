"""
Unit tests for clustering pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from geosuite.ml.clustering import (
    FaciesClusterer,
    cluster_facies,
    find_optimal_clusters,
)


class TestFaciesClusterer:
    """Tests for FaciesClusterer class."""
    
    def test_kmeans_basic(self):
        """Test basic KMeans clustering."""
        X = np.random.rand(100, 4)
        clusterer = FaciesClusterer(method='kmeans', n_clusters=3, random_state=42)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 100
        assert len(set(labels)) == 3
        assert np.all(labels >= 0)
    
    def test_kmeans_pandas(self):
        """Test KMeans with pandas DataFrame."""
        df = pd.DataFrame({
            'GR': np.random.rand(50),
            'RHOB': np.random.rand(50),
            'NPHI': np.random.rand(50),
        })
        clusterer = FaciesClusterer(method='kmeans', n_clusters=2, random_state=42)
        labels = clusterer.fit_predict(df)
        
        assert len(labels) == 50
        assert clusterer.feature_names == ['GR', 'RHOB', 'NPHI']
    
    def test_dbscan_basic(self):
        """Test DBSCAN clustering."""
        X = np.random.rand(100, 4)
        clusterer = FaciesClusterer(method='dbscan', eps=0.5, min_samples=5, random_state=42)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 100
        assert -1 in labels or len(set(labels)) > 0
    
    def test_hierarchical_basic(self):
        """Test hierarchical clustering."""
        X = np.random.rand(50, 4)
        clusterer = FaciesClusterer(method='hierarchical', n_clusters=3, random_state=42)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 50
        assert len(set(labels)) == 3
    
    def test_scale_features(self):
        """Test with and without feature scaling."""
        X = np.random.rand(50, 4) * 100
        
        clusterer_scaled = FaciesClusterer(method='kmeans', n_clusters=2, scale_features=True, random_state=42)
        labels_scaled = clusterer_scaled.fit_predict(X)
        
        clusterer_unscaled = FaciesClusterer(method='kmeans', n_clusters=2, scale_features=False, random_state=42)
        labels_unscaled = clusterer_unscaled.fit_predict(X)
        
        assert len(labels_scaled) == len(labels_unscaled)
    
    def test_invalid_method(self):
        """Test with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            FaciesClusterer(method='invalid', n_clusters=3)
    
    def test_missing_n_clusters(self):
        """Test error when n_clusters not provided for kmeans."""
        with pytest.raises(ValueError, match="n_clusters must be specified"):
            FaciesClusterer(method='kmeans', random_state=42)
    
    def test_empty_data(self):
        """Test with empty data."""
        clusterer = FaciesClusterer(method='kmeans', n_clusters=2, random_state=42)
        with pytest.raises(ValueError, match="must not be empty"):
            clusterer.fit(np.array([]).reshape(0, 4))
    
    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        clusterer = FaciesClusterer(method='kmeans', n_clusters=2, random_state=42)
        with pytest.raises(ValueError, match="must be fitted"):
            clusterer.predict(np.random.rand(10, 4))
    
    def test_fit_then_predict(self):
        """Test fit then predict workflow."""
        X_train = np.random.rand(100, 4)
        X_test = np.random.rand(20, 4)
        
        clusterer = FaciesClusterer(method='kmeans', n_clusters=3, random_state=42)
        clusterer.fit(X_train)
        labels = clusterer.predict(X_test)
        
        assert len(labels) == 20
        assert np.all(labels >= 0)
        assert np.all(labels < 3)


class TestClusterFacies:
    """Tests for cluster_facies convenience function."""
    
    def test_basic_clustering(self):
        """Test basic facies clustering."""
        df = pd.DataFrame({
            'GR': np.random.rand(50),
            'RHOB': np.random.rand(50),
            'NPHI': np.random.rand(50),
            'depth': np.arange(50),
        })
        
        labels = cluster_facies(
            df,
            feature_cols=['GR', 'RHOB', 'NPHI'],
            method='kmeans',
            n_clusters=3,
            random_state=42
        )
        
        assert len(labels) == 50
        assert isinstance(labels, pd.Series)
        assert labels.name == 'facies_cluster'
        assert labels.index.equals(df.index)
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering."""
        df = pd.DataFrame({
            'GR': np.random.rand(50),
            'RHOB': np.random.rand(50),
        })
        
        labels = cluster_facies(
            df,
            feature_cols=['GR', 'RHOB'],
            method='dbscan',
            eps=0.5,
            min_samples=5,
            random_state=42
        )
        
        assert len(labels) == 50


class TestFindOptimalClusters:
    """Tests for find_optimal_clusters function."""
    
    def test_basic_elbow(self):
        """Test basic elbow method."""
        X = np.random.rand(100, 4)
        result = find_optimal_clusters(X, method='kmeans', max_clusters=8, random_state=42)
        
        assert 'n_clusters' in result
        assert 'inertias' in result
        assert 'optimal_n' in result
        assert len(result['n_clusters']) == 7
        assert len(result['inertias']) == 7
        assert 2 <= result['optimal_n'] <= 8
    
    def test_pandas_input(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({
            'GR': np.random.rand(50),
            'RHOB': np.random.rand(50),
        })
        result = find_optimal_clusters(df, method='kmeans', max_clusters=5, random_state=42)
        
        assert 'optimal_n' in result
    
    def test_invalid_method(self):
        """Test with invalid method."""
        X = np.random.rand(50, 4)
        with pytest.raises(ValueError, match="currently only supports"):
            find_optimal_clusters(X, method='dbscan', max_clusters=5)

