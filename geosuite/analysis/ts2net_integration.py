"""
Integration with ts2net for time series to network analysis.

Converts well log time series into network representations for structural
analysis, pattern detection, and multi-well comparison.
"""
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import ts2net
try:
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork, build_network
    from ts2net.multivariate import ts_dist, net_knn, net_enn
    TS2NET_AVAILABLE = True
except ImportError:
    TS2NET_AVAILABLE = False
    logger.warning(
        "ts2net not available. Time series to network analysis requires ts2net. "
        "Install with: pip install ts2net"
    )
    HVG = None
    NVG = None
    RecurrenceNetwork = None
    TransitionNetwork = None
    build_network = None
    ts_dist = None
    net_knn = None
    net_enn = None


class WellLogNetworkAnalyzer:
    """
    Analyzer for converting well log time series to network representations.
    
    Provides methods for analyzing well log data using visibility graphs,
    recurrence networks, and transition networks to extract structural
    features and detect patterns.
    
    Example:
        >>> from geosuite.analysis import WellLogNetworkAnalyzer
        >>> 
        >>> analyzer = WellLogNetworkAnalyzer(method='hvg')
        >>> result = analyzer.analyze(df['GR'].values)
        >>> print(f"Network edges: {result['n_edges']}")
        >>> print(f"Average degree: {result['avg_degree']}")
    """
    
    def __init__(
        self,
        method: str = 'hvg',
        depth_col: str = 'DEPTH',
        **network_params
    ):
        """
        Initialize well log network analyzer.
        
        Parameters
        ----------
        method : str, default 'hvg'
            Network construction method:
            - 'hvg': Horizontal Visibility Graph
            - 'nvg': Natural Visibility Graph
            - 'recurrence': Recurrence Network
            - 'transition': Transition Network
        depth_col : str, default 'DEPTH'
            Column name for depth (used when processing DataFrames)
        **network_params
            Additional parameters passed to network builder
        """
        if not TS2NET_AVAILABLE:
            raise ImportError(
                "ts2net is required for network analysis. "
                "Install with: pip install ts2net"
            )
        
        self.method = method.lower()
        self.depth_col = depth_col
        self.network_params = network_params
        self.network = None
        
        method_map = {
            'hvg': HVG,
            'nvg': NVG,
            'recurrence': RecurrenceNetwork,
            'transition': TransitionNetwork
        }
        
        if self.method not in method_map:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from: {', '.join(method_map.keys())}"
            )
        
        self.network_class = method_map[self.method]
    
    def analyze(
        self,
        log_values: Union[np.ndarray, pd.Series, pd.DataFrame],
        log_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze well log time series as a network.
        
        Parameters
        ----------
        log_values : np.ndarray, pd.Series, or pd.DataFrame
            Well log values (time series)
        log_col : str, optional
            Column name if log_values is a DataFrame
            
        Returns
        -------
        dict
            Dictionary containing network statistics and features:
            - 'n_nodes': Number of nodes
            - 'n_edges': Number of edges
            - 'avg_degree': Average node degree
            - 'density': Network density
            - 'network': Network object
            - 'degree_sequence': Degree sequence array
        """
        # Extract time series
        if isinstance(log_values, pd.DataFrame):
            if log_col is None:
                raise ValueError("log_col must be specified when log_values is a DataFrame")
            ts = log_values[log_col].values
        elif isinstance(log_values, pd.Series):
            ts = log_values.values
        else:
            ts = np.asarray(log_values)
        
        # Remove NaN values
        valid_mask = ~np.isnan(ts)
        ts_clean = ts[valid_mask]
        
        if len(ts_clean) < 10:
            raise ValueError("Time series must have at least 10 valid points")
        
        # Build network
        if self.method == 'recurrence':
            # Recurrence networks need embedding parameters
            default_params = {
                'm': 3,
                'tau': 1,
                'rule': 'knn',
                'k': 5
            }
            default_params.update(self.network_params)
            self.network = self.network_class(**default_params)
        elif self.method == 'transition':
            # Transition networks need symbolizer parameters
            default_params = {
                'symbolizer': 'ordinal',
                'order': 3
            }
            default_params.update(self.network_params)
            self.network = self.network_class(**default_params)
        else:
            # Visibility graphs
            default_params = {
                'weighted': False,
                'output': 'edges'
            }
            default_params.update(self.network_params)
            self.network = self.network_class(**default_params)
        
        self.network.build(ts_clean)
        
        # Extract features
        n_nodes = self.network.n_nodes
        n_edges = self.network.n_edges
        degree_seq = self.network.degree_sequence()
        avg_degree = np.mean(degree_seq) if len(degree_seq) > 0 else 0.0
        max_degree = np.max(degree_seq) if len(degree_seq) > 0 else 0
        min_degree = np.min(degree_seq) if len(degree_seq) > 0 else 0
        
        # Calculate density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0.0
        
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'max_degree': int(max_degree),
            'min_degree': int(min_degree),
            'density': density,
            'network': self.network,
            'degree_sequence': degree_seq,
            'method': self.method
        }
    
    def compare_with_reference(
        self,
        log_values: Union[np.ndarray, pd.Series],
        reference_values: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Compare a well log with a reference using network features.
        
        Parameters
        ----------
        log_values : np.ndarray or pd.Series
            Well log values to analyze
        reference_values : np.ndarray or pd.Series
            Reference well log values
            
        Returns
        -------
        dict
            Comparison metrics including network feature differences
        """
        result_log = self.analyze(log_values)
        result_ref = self.analyze(reference_values)
        
        return {
            'log_features': result_log,
            'reference_features': result_ref,
            'degree_diff': abs(result_log['avg_degree'] - result_ref['avg_degree']),
            'density_diff': abs(result_log['density'] - result_ref['density']),
            'edge_ratio': result_log['n_edges'] / result_ref['n_edges'] if result_ref['n_edges'] > 0 else np.inf
        }


def analyze_well_log_network(
    df: pd.DataFrame,
    log_col: str,
    method: str = 'hvg',
    depth_col: str = 'DEPTH',
    **network_params
) -> Dict[str, Any]:
    """
    Analyze a well log column as a network.
    
    Convenience function for quick network analysis of well log data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Well log DataFrame
    log_col : str
        Column name of log to analyze
    method : str, default 'hvg'
        Network method ('hvg', 'nvg', 'recurrence', 'transition')
    depth_col : str, default 'DEPTH'
        Column name for depth
    **network_params
        Additional parameters for network builder
        
    Returns
    -------
    dict
        Network analysis results
    """
    analyzer = WellLogNetworkAnalyzer(method=method, depth_col=depth_col, **network_params)
    return analyzer.analyze(df, log_col=log_col)


def compare_wells_network(
    wells_data: List[Dict[str, Any]],
    log_col: str,
    method: str = 'hvg',
    distance_metric: str = 'dtw',
    **network_params
) -> Dict[str, Any]:
    """
    Compare multiple wells using network analysis.
    
    Creates a network where nodes are wells and edges represent similarity
    based on network features extracted from their well logs.
    
    Parameters
    ----------
    wells_data : list of dict
        List of dictionaries, each containing:
        - 'df': DataFrame with well log data
        - 'name': Well name (optional)
    log_col : str
        Column name of log to analyze
    method : str, default 'hvg'
        Network method for individual well analysis
    distance_metric : str, default 'dtw'
        Distance metric for well comparison ('dtw', 'correlation', 'nmi', etc.)
    **network_params
        Additional parameters for network builder
        
    Returns
    -------
    dict
        Comparison results including:
        - 'well_networks': Network features for each well
        - 'well_similarity_network': Network of well similarities
        - 'similarity_matrix': Pairwise similarity matrix
    """
    if not TS2NET_AVAILABLE:
        raise ImportError("ts2net is required for well comparison")
    
    # Extract time series from each well
    well_series = []
    well_names = []
    
    for well_info in wells_data:
        df = well_info['df']
        name = well_info.get('name', f'Well_{len(well_names)}')
        
        if log_col not in df.columns:
            logger.warning(f"Log column '{log_col}' not found in well {name}, skipping")
            continue
        
        ts = df[log_col].values
        ts_clean = ts[~np.isnan(ts)]
        
        if len(ts_clean) < 10:
            logger.warning(f"Well {name} has insufficient data, skipping")
            continue
        
        well_series.append(ts_clean)
        well_names.append(name)
    
    if len(well_series) < 2:
        raise ValueError("Need at least 2 wells for comparison")
    
    # Analyze each well individually
    analyzer = WellLogNetworkAnalyzer(method=method, **network_params)
    well_networks = {}
    
    for name, ts in zip(well_names, well_series):
        well_networks[name] = analyzer.analyze(ts)
    
    # Create multivariate network for well comparison
    # Pad series to same length for distance calculation
    max_len = max(len(ts) for ts in well_series)
    X = np.array([np.pad(ts, (0, max_len - len(ts)), constant_values=np.nan) 
                  for ts in well_series])
    
    # Calculate distance matrix
    D = ts_dist(X, method=distance_metric, n_jobs=-1)
    
    # Build similarity network
    G = net_knn(D, k=min(3, len(well_names) - 1))
    
    # Create similarity matrix (invert distance)
    similarity_matrix = 1 / (1 + D)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return {
        'well_networks': well_networks,
        'well_similarity_network': G,
        'similarity_matrix': similarity_matrix,
        'well_names': well_names,
        'distance_matrix': D
    }


def detect_formation_boundaries_network(
    df: pd.DataFrame,
    log_col: str,
    window_size: int = 100,
    step_size: int = 50,
    method: str = 'hvg',
    threshold: float = 0.3,
    depth_col: str = 'DEPTH',
    **network_params
) -> np.ndarray:
    """
    Detect formation boundaries using sliding window network analysis.
    
    Uses network features (e.g., density, average degree) to identify
    changes in well log structure that may indicate formation boundaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Well log DataFrame
    log_col : str
        Column name of log to analyze
    window_size : int, default 100
        Size of sliding window
    step_size : int, default 50
        Step size for sliding window
    method : str, default 'hvg'
        Network method
    threshold : float, default 0.3
        Threshold for detecting significant changes in network features
    depth_col : str, default 'DEPTH'
        Column name for depth
    **network_params
        Additional parameters for network builder
        
    Returns
    -------
    np.ndarray
        Array of depth indices where boundaries are detected
    """
    if log_col not in df.columns:
        raise ValueError(f"Log column '{log_col}' not found")
    
    log_values = df[log_col].values
    depth_values = df[depth_col].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(log_values)
    log_clean = log_values[valid_mask]
    depth_clean = depth_values[valid_mask]
    
    if len(log_clean) < window_size:
        logger.warning("Time series too short for sliding window analysis")
        return np.array([])
    
    analyzer = WellLogNetworkAnalyzer(method=method, **network_params)
    
    # Sliding window analysis
    features = []
    window_centers = []
    
    for i in range(0, len(log_clean) - window_size + 1, step_size):
        window_data = log_clean[i:i + window_size]
        
        try:
            result = analyzer.analyze(window_data)
            features.append({
                'density': result['density'],
                'avg_degree': result['avg_degree'],
                'n_edges': result['n_edges']
            })
            window_centers.append(i + window_size // 2)
        except Exception as e:
            logger.warning(f"Failed to analyze window at index {i}: {e}")
            continue
    
    if len(features) < 2:
        return np.array([])
    
    # Detect changes in network features
    densities = np.array([f['density'] for f in features])
    avg_degrees = np.array([f['avg_degree'] for f in features])
    
    # Calculate change points based on feature differences
    density_diff = np.abs(np.diff(densities))
    degree_diff = np.abs(np.diff(avg_degrees))
    
    # Normalize differences
    density_diff_norm = density_diff / (np.max(density_diff) + 1e-10)
    degree_diff_norm = degree_diff / (np.max(degree_diff) + 1e-10)
    
    # Combined change score
    change_scores = (density_diff_norm + degree_diff_norm) / 2
    
    # Find boundaries above threshold
    boundary_indices = np.where(change_scores > threshold)[0]
    
    # Convert window centers to depth indices
    if len(boundary_indices) > 0:
        depth_boundaries = depth_clean[window_centers[boundary_indices[0]]]
        return np.array([depth_boundaries]) if np.isscalar(depth_boundaries) else depth_clean[window_centers[boundary_indices]]
    
    return np.array([])


def extract_network_features(
    df: pd.DataFrame,
    log_cols: List[str],
    method: str = 'hvg',
    depth_col: str = 'DEPTH',
    **network_params
) -> pd.DataFrame:
    """
    Extract network features from multiple well log columns.
    
    Creates a feature matrix where each row is a depth window and columns
    are network features extracted from each log.
    
    Parameters
    ----------
    df : pd.DataFrame
        Well log DataFrame
    log_cols : list of str
        List of log column names to analyze
    method : str, default 'hvg'
        Network method
    depth_col : str, default 'DEPTH'
        Column name for depth
    **network_params
        Additional parameters for network builder
        
    Returns
    -------
    pd.DataFrame
        DataFrame with network features for each log column
    """
    analyzer = WellLogNetworkAnalyzer(method=method, depth_col=depth_col, **network_params)
    
    features_dict = {}
    
    for log_col in log_cols:
        if log_col not in df.columns:
            logger.warning(f"Log column '{log_col}' not found, skipping")
            continue
        
        try:
            result = analyzer.analyze(df, log_col=log_col)
            features_dict[f'{log_col}_n_nodes'] = result['n_nodes']
            features_dict[f'{log_col}_n_edges'] = result['n_edges']
            features_dict[f'{log_col}_avg_degree'] = result['avg_degree']
            features_dict[f'{log_col}_density'] = result['density']
            features_dict[f'{log_col}_max_degree'] = result['max_degree']
            features_dict[f'{log_col}_min_degree'] = result['min_degree']
        except Exception as e:
            logger.warning(f"Failed to analyze {log_col}: {e}")
            continue
    
    return pd.DataFrame([features_dict])

