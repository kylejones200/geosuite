"""
Advanced stratigraphy tools: ML-based segmentation, multi-log detection, and correlation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, List, Optional, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def ml_segment_timeseries(
    log_data: Union[np.ndarray, pd.DataFrame],
    n_segments: Optional[int] = None,
    method: str = 'kmeans',
    features: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Segment well log time series using machine learning.
    
    Uses clustering or segmentation algorithms to identify distinct stratigraphic units.
    
    Args:
        log_data: Well log data (array or DataFrame with multiple logs)
        n_segments: Number of segments (auto-determined if None)
        method: Segmentation method ('kmeans', 'pca_kmeans', 'hierarchical')
        features: Feature columns to use (if DataFrame)
        random_state: Random seed
        
    Returns:
        Dictionary with segment labels, boundaries, and statistics
    """
    if isinstance(log_data, pd.DataFrame):
        if features is None:
            features = [col for col in log_data.columns if col not in ['depth', 'DEPTH', 'depth_m']]
        X = log_data[features].values
    else:
        X = np.asarray(log_data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    
    if len(X) == 0:
        raise ValueError("Log data must not be empty")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model_configs = {
        'kmeans': lambda n: KMeans(n_clusters=n, random_state=random_state, n_init=10),
        'pca_kmeans': lambda n: _pca_kmeans_segmentation(X_scaled, n, random_state),
        'hierarchical': lambda n: _hierarchical_segmentation(X_scaled, n),
    }
    
    if method not in model_configs:
        raise ValueError(f"Unknown method: {method}. Choose: {', '.join(model_configs.keys())}")
    
    if n_segments is None:
        n_segments = _estimate_optimal_segments(X_scaled)
    
    model = model_configs[method](n_segments)
    
    if method == 'pca_kmeans':
        labels = model
    elif method == 'hierarchical':
        labels = model
    else:
        labels = model.fit_predict(X_scaled)
    
    # Find segment boundaries
    boundaries = _find_segment_boundaries(labels)
    
    # Calculate segment statistics
    segment_stats = _calculate_segment_statistics(X, labels, boundaries)
    
    return {
        'labels': labels,
        'boundaries': boundaries,
        'n_segments': n_segments,
        'statistics': segment_stats
    }


def _pca_kmeans_segmentation(X: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Segment using PCA + KMeans."""
    pca = PCA(n_components=min(3, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    return labels


def _hierarchical_segmentation(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Segment using hierarchical clustering."""
    from sklearn.cluster import AgglomerativeClustering
    
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    
    return labels


def _estimate_optimal_segments(X: np.ndarray, max_segments: int = 20) -> int:
    """Estimate optimal number of segments using elbow method."""
    inertias = []
    n_range = range(2, min(max_segments + 1, len(X) // 10))
    
    for n in n_range:
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    if len(inertias) < 2:
        return 5
    
    # Find elbow
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 2
    
    return min(elbow_idx, max_segments)


def _find_segment_boundaries(labels: np.ndarray) -> np.ndarray:
    """Find boundaries between segments."""
    boundaries = np.where(np.diff(labels) != 0)[0] + 1
    return boundaries


def _calculate_segment_statistics(X: np.ndarray, labels: np.ndarray, boundaries: np.ndarray) -> Dict:
    """Calculate statistics for each segment."""
    stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        segment_data = X[mask]
        
        stats[int(label)] = {
            'mean': np.mean(segment_data, axis=0),
            'std': np.std(segment_data, axis=0),
            'size': np.sum(mask),
            'start_idx': np.where(mask)[0][0] if np.any(mask) else 0,
            'end_idx': np.where(mask)[0][-1] if np.any(mask) else len(labels) - 1,
        }
    
    return stats


def detect_multi_log_boundaries(
    log_data: pd.DataFrame,
    log_columns: List[str],
    depth_column: str = 'depth',
    method: str = 'consensus',
    tolerance: float = 5.0,
    min_agreement: int = 2
) -> Dict[str, np.ndarray]:
    """
    Detect formation boundaries using multiple well logs.
    
    Combines change-point detection from multiple logs to find robust boundaries.
    
    Args:
        log_data: DataFrame with well log data
        log_columns: List of log column names to analyze
        depth_column: Name of depth column
        method: Detection method ('consensus', 'weighted', 'majority')
        tolerance: Depth tolerance for matching boundaries (same units as depth)
        min_agreement: Minimum number of logs that must agree on a boundary
        
    Returns:
        Dictionary with consensus boundaries, individual detections, and confidence
    """
    if depth_column not in log_data.columns:
        raise ValueError(f"Depth column '{depth_column}' not found in data")
    
    depth = log_data[depth_column].values
    
    from geosuite.stratigraphy.changepoint import detect_pelt, preprocess_log
    
    individual_detections = {}
    all_boundaries = []
    
    for log_col in log_columns:
        if log_col not in log_data.columns:
            logger.warning(f"Log column '{log_col}' not found, skipping")
            continue
        
        log_values = log_data[log_col].values
        
        try:
            log_processed = preprocess_log(log_values)
            boundaries = detect_pelt(log_processed, penalty=None)
            boundary_depths = depth[boundaries]
            
            individual_detections[log_col] = {
                'indices': boundaries,
                'depths': boundary_depths
            }
            all_boundaries.extend(boundary_depths.tolist())
        except Exception as e:
            logger.warning(f"Failed to detect boundaries in {log_col}: {e}")
    
    if len(all_boundaries) == 0:
        return {
            'consensus_boundaries': np.array([]),
            'consensus_depths': np.array([]),
            'individual_detections': individual_detections,
            'confidence': np.array([])
        }
    
    # Find consensus boundaries
    all_boundaries = np.array(all_boundaries)
    consensus_boundaries, confidence = _find_consensus_boundaries(
        all_boundaries, depth, tolerance, min_agreement, method
    )
    
    return {
        'consensus_boundaries': consensus_boundaries,
        'consensus_depths': depth[consensus_boundaries] if len(consensus_boundaries) > 0 else np.array([]),
        'individual_detections': individual_detections,
        'confidence': confidence
    }


def _find_consensus_boundaries(
    all_boundaries: np.ndarray,
    depth: np.ndarray,
    tolerance: float,
    min_agreement: int,
    method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Find consensus boundaries from multiple detections."""
    if len(all_boundaries) == 0:
        return np.array([]), np.array([])
    
    all_boundaries = np.sort(all_boundaries)
    
    # Cluster nearby boundaries
    clusters = []
    current_cluster = [all_boundaries[0]]
    
    for boundary in all_boundaries[1:]:
        if boundary - current_cluster[-1] <= tolerance:
            current_cluster.append(boundary)
        else:
            clusters.append(current_cluster)
            current_cluster = [boundary]
    clusters.append(current_cluster)
    
    # Find clusters with sufficient agreement
    consensus = []
    confidence_scores = []
    
    for cluster in clusters:
        if len(cluster) >= min_agreement:
            if method == 'weighted':
                # Weight by number of detections
                consensus_depth = np.average(cluster, weights=[1.0] * len(cluster))
            else:
                # Simple mean
                consensus_depth = np.mean(cluster)
            
            # Find nearest depth index
            nearest_idx = np.argmin(np.abs(depth - consensus_depth))
            consensus.append(nearest_idx)
            confidence_scores.append(len(cluster) / len(all_boundaries))
    
    return np.array(consensus), np.array(confidence_scores)


def correlate_formations(
    well_data: Dict[str, pd.DataFrame],
    depth_columns: Optional[Dict[str, str]] = None,
    log_columns: Optional[Dict[str, List[str]]] = None,
    reference_well: Optional[str] = None,
    method: str = 'dynamic_time_warping'
) -> Dict[str, Dict]:
    """
    Correlate formations across multiple wells.
    
    Matches formation boundaries between wells using pattern matching.
    
    Args:
        well_data: Dictionary of {well_name: DataFrame} with log data
        depth_columns: Dictionary of {well_name: depth_column_name} (default 'depth')
        log_columns: Dictionary of {well_name: [log_columns]} (uses all numeric columns if None)
        reference_well: Well to use as reference (first well if None)
        method: Correlation method ('dynamic_time_warping', 'cross_correlation', 'feature_matching')
        
    Returns:
        Dictionary with correlation results and matched boundaries
    """
    if len(well_data) < 2:
        raise ValueError("At least two wells required for correlation")
    
    depth_columns = depth_columns or {name: 'depth' for name in well_data.keys()}
    log_columns = log_columns or {}
    
    if reference_well is None:
        reference_well = list(well_data.keys())[0]
    
    if reference_well not in well_data:
        raise ValueError(f"Reference well '{reference_well}' not found")
    
    reference_df = well_data[reference_well]
    reference_depth_col = depth_columns.get(reference_well, 'depth')
    reference_logs = log_columns.get(reference_well, 
                                     [col for col in reference_df.columns 
                                      if col != reference_depth_col and pd.api.types.is_numeric_dtype(reference_df[col])])
    
    correlations = {}
    
    for well_name, well_df in well_data.items():
        if well_name == reference_well:
            continue
        
        depth_col = depth_columns.get(well_name, 'depth')
        logs = log_columns.get(well_name,
                             [col for col in well_df.columns 
                              if col != depth_col and pd.api.types.is_numeric_dtype(well_df[col])])
        
        try:
            if method == 'dynamic_time_warping':
                correlation = _correlate_dtw(reference_df, well_df, reference_logs, logs, 
                                           reference_depth_col, depth_col)
            elif method == 'cross_correlation':
                correlation = _correlate_cross_correlation(reference_df, well_df, reference_logs, logs,
                                                          reference_depth_col, depth_col)
            else:
                correlation = _correlate_feature_matching(reference_df, well_df, reference_logs, logs,
                                                         reference_depth_col, depth_col)
            
            correlations[well_name] = correlation
        except Exception as e:
            logger.warning(f"Failed to correlate {well_name}: {e}")
            correlations[well_name] = {'error': str(e)}
    
    return {
        'reference_well': reference_well,
        'correlations': correlations,
        'method': method
    }


def _correlate_dtw(
    ref_df: pd.DataFrame,
    well_df: pd.DataFrame,
    ref_logs: List[str],
    logs: List[str],
    ref_depth_col: str,
    depth_col: str
) -> Dict:
    """Correlate using dynamic time warping (simplified)."""
    # Use common logs if available
    common_logs = [log for log in ref_logs if log in logs]
    if len(common_logs) == 0:
        common_logs = ref_logs[:1] if ref_logs else []
    
    if len(common_logs) == 0:
        return {'error': 'No common logs found'}
    
    ref_log = ref_df[common_logs[0]].values
    well_log = well_df[common_logs[0]].values
    
    # Simplified DTW: find best offset
    max_offset = min(len(ref_log), len(well_log)) // 10
    best_offset = 0
    best_correlation = -1
    
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            ref_slice = ref_log[offset:]
            well_slice = well_log[:len(ref_slice)]
        else:
            ref_slice = ref_log[:len(well_log) + offset]
            well_slice = well_log[-offset:]
        
        if len(ref_slice) > 10 and len(well_slice) > 10:
            corr = np.corrcoef(ref_slice, well_slice)[0, 1]
            if not np.isnan(corr) and corr > best_correlation:
                best_correlation = corr
                best_offset = offset
    
    return {
        'offset': best_offset,
        'correlation': best_correlation,
        'method': 'dtw_simplified'
    }


def _correlate_cross_correlation(
    ref_df: pd.DataFrame,
    well_df: pd.DataFrame,
    ref_logs: List[str],
    logs: List[str],
    ref_depth_col: str,
    depth_col: str
) -> Dict:
    """Correlate using cross-correlation."""
    common_logs = [log for log in ref_logs if log in logs]
    if len(common_logs) == 0:
        return {'error': 'No common logs found'}
    
    ref_log = ref_df[common_logs[0]].values
    well_log = well_df[common_logs[0]].values
    
    # Normalize
    ref_log = (ref_log - np.mean(ref_log)) / (np.std(ref_log) + 1e-10)
    well_log = (well_log - np.mean(well_log)) / (np.std(well_log) + 1e-10)
    
    # Cross-correlation
    min_len = min(len(ref_log), len(well_log))
    ref_log = ref_log[:min_len]
    well_log = well_log[:min_len]
    
    correlation = np.corrcoef(ref_log, well_log)[0, 1]
    
    return {
        'correlation': correlation if not np.isnan(correlation) else 0.0,
        'method': 'cross_correlation'
    }


def _correlate_feature_matching(
    ref_df: pd.DataFrame,
    well_df: pd.DataFrame,
    ref_logs: List[str],
    logs: List[str],
    ref_depth_col: str,
    depth_col: str
) -> Dict:
    """Correlate using feature matching."""
    common_logs = [log for log in ref_logs if log in logs]
    if len(common_logs) == 0:
        return {'error': 'No common logs found'}
    
    # Use first common log
    ref_log = ref_df[common_logs[0]].values
    well_log = well_df[common_logs[0]].values
    
    # Extract features (mean, std, min, max in windows)
    window_size = min(50, len(ref_log) // 10, len(well_log) // 10)
    
    ref_features = _extract_features(ref_log, window_size)
    well_features = _extract_features(well_log, window_size)
    
    # Match features
    if len(ref_features) > 0 and len(well_features) > 0:
        correlation = np.corrcoef(ref_features.flatten(), well_features.flatten())[0, 1]
    else:
        correlation = 0.0
    
    return {
        'correlation': correlation if not np.isnan(correlation) else 0.0,
        'method': 'feature_matching'
    }


def _extract_features(log: np.ndarray, window_size: int) -> np.ndarray:
    """Extract statistical features from log in windows."""
    n_windows = len(log) // window_size
    if n_windows == 0:
        return np.array([])
    
    features = []
    for i in range(n_windows):
        window = log[i * window_size:(i + 1) * window_size]
        features.append([np.mean(window), np.std(window), np.min(window), np.max(window)])
    
    return np.array(features)

