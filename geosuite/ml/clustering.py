"""
Clustering pipelines for facies grouping.

Unsupervised learning methods for identifying facies groups from well log data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.pipeline import Pipeline

from ..base.estimators import BaseEstimator

logger = logging.getLogger(__name__)


class FaciesClusterer(BaseEstimator):
    """
    Clustering pipeline for facies identification from well log data.
    
    Supports multiple clustering algorithms and provides a consistent API
    for facies grouping workflows.
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        scale_features: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the facies clusterer.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (required for kmeans/hierarchical, ignored for dbscan)
            scale_features: Whether to scale features before clustering
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters passed to clustering algorithm
        """
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.scale_features = scale_features
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self.feature_names = None
        self.cluster_labels_ = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the clustering model based on method."""
        model_configs = {
            'kmeans': lambda: KMeans(
                n_clusters=self.n_clusters or 5,
                random_state=self.random_state,
                n_init=10,
                **self.kwargs
            ),
            'dbscan': lambda: DBSCAN(
                eps=self.kwargs.get('eps', 0.5),
                min_samples=self.kwargs.get('min_samples', 5),
                **{k: v for k, v in self.kwargs.items() if k not in ['eps', 'min_samples']}
            ),
            'hierarchical': lambda: AgglomerativeClustering(
                n_clusters=self.n_clusters or 5,
                linkage=self.kwargs.get('linkage', 'ward'),
                **{k: v for k, v in self.kwargs.items() if k != 'linkage'}
            ),
        }
        
        if self.method not in model_configs:
            raise ValueError(f"Unknown method: {self.method}. Choose: {', '.join(model_configs.keys())}")
        
        if self.method in ['kmeans', 'hierarchical'] and self.n_clusters is None:
            raise ValueError(f"n_clusters must be specified for {self.method}")
        
        self.model = model_configs[self.method]()
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'FaciesClusterer':
        """
        Fit the clustering model to data.
        
        Args:
            X: Feature array or DataFrame
            y: Ignored (clustering is unsupervised), optional
            
        Returns:
            self
        """
        # For clustering, y is not used but we accept it for BaseEstimator compatibility
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = np.asarray(X)
            self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        
        if len(X_array) == 0:
            raise ValueError("Input data must not be empty")
        
        if self.scale_features:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = X_array
        
        self.model.fit(X_scaled)
        self.cluster_labels_ = self.model.labels_
        
        n_clusters = len(set(self.cluster_labels_)) - (1 if -1 in self.cluster_labels_ else 0)
        logger.info(f"Fitted {self.method} clustering with {n_clusters} clusters")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Feature array or DataFrame
            
        Returns:
            Cluster labels array
        """
        if self.model is None or self.cluster_labels_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        if self.scale_features:
            if self.scaler is None:
                raise ValueError("Scaler must be fitted before prediction")
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array
        
        if self.method == 'dbscan':
            return self.model.fit_predict(X_scaled)
        else:
            return self.model.predict(X_scaled)
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Feature array or DataFrame
            
        Returns:
            Cluster labels array
        """
        return self.fit(X).cluster_labels_


def cluster_facies(
    df: pd.DataFrame,
    feature_cols: List[str],
    method: str = 'kmeans',
    n_clusters: Optional[int] = None,
    scale_features: bool = True,
    random_state: int = 42,
    **kwargs
) -> pd.Series:
    """
    Cluster facies from well log data.
    
    Convenience function for quick facies clustering.
    
    Args:
        df: DataFrame with well log data
        feature_cols: List of column names to use as features
        method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: Number of clusters (required for kmeans/hierarchical)
        scale_features: Whether to scale features
        random_state: Random seed
        **kwargs: Additional clustering parameters
        
    Returns:
        Series with cluster labels
    """
    clusterer = FaciesClusterer(
        method=method,
        n_clusters=n_clusters,
        scale_features=scale_features,
        random_state=random_state,
        **kwargs
    )
    
    labels = clusterer.fit_predict(df[feature_cols])
    
    return pd.Series(labels, index=df.index, name='facies_cluster')


def find_optimal_clusters(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = 'kmeans',
    max_clusters: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method (for kmeans).
    
    Args:
        X: Feature array or DataFrame
        method: Clustering method (currently only 'kmeans' supported)
        max_clusters: Maximum number of clusters to test
        random_state: Random seed
        
    Returns:
        Dictionary with cluster counts and inertia/silhouette scores
    """
    if method != 'kmeans':
        raise ValueError("Optimal cluster finding currently only supports kmeans")
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    inertias = []
    n_clusters_range = range(2, max_clusters + 1)
    
    for n in n_clusters_range:
        kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    return {
        'n_clusters': list(n_clusters_range),
        'inertias': inertias,
        'optimal_n': _find_elbow(inertias, n_clusters_range)
    }


def _find_elbow(inertias: List[float], n_clusters: List[int]) -> int:
    """
    Find elbow point in inertia curve.
    
    Uses the point of maximum curvature.
    """
    if len(inertias) < 3:
        return n_clusters[len(inertias) // 2]
    
    inertias = np.array(inertias)
    n_clusters = np.array(n_clusters)
    
    first_point = np.array([n_clusters[0], inertias[0]])
    last_point = np.array([n_clusters[-1], inertias[-1]])
    
    distances = []
    for i in range(len(n_clusters)):
        point = np.array([n_clusters[i], inertias[i]])
        d = np.abs(np.cross(last_point - first_point, first_point - point)) / np.linalg.norm(last_point - first_point)
        distances.append(d)
    
    elbow_idx = np.argmax(distances)
    return int(n_clusters[elbow_idx])

