"""
Cross-validation schemes for geoscience data.

Provides well-to-well and spatial cross-validation that respects
geological correlation and prevents data leakage.
"""
import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class WellBasedKFold(BaseCrossValidator):
    """
    K-Fold cross-validation that groups by well to prevent data leakage.
    
    This ensures that data from the same well stays together in train/test splits,
    preventing overfitting due to spatial correlation within wells.
    
    Example:
        >>> from geosuite.ml import WellBasedKFold
        >>> from geosuite.ml import PermeabilityPredictor
        >>> 
        >>> cv = WellBasedKFold(n_splits=5, well_col='WELL')
        >>> predictor = PermeabilityPredictor()
        >>> scores = cross_val_score(predictor, X, y, cv=cv)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        well_col: str = 'WELL',
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize well-based K-Fold cross-validator.
        
        Parameters
        ----------
        n_splits : int, default 5
            Number of folds
        well_col : str, default 'WELL'
            Column name containing well identifiers
        shuffle : bool, default False
            Whether to shuffle wells before splitting
        random_state : int, optional
            Random seed for shuffling
        """
        self.n_splits = n_splits
        self.well_col = well_col
        self.shuffle = shuffle
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ):
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data
        y : np.ndarray or pd.Series, optional
            Target values (unused)
        groups : np.ndarray, optional
            Group labels (unused, well_col is used instead)
            
        Yields
        ------
        train_indices : np.ndarray
            Training set indices
        test_indices : np.ndarray
            Test set indices
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            raise ValueError(
                f"X must be a DataFrame with '{self.well_col}' column"
            )
        
        df = X.copy()
        
        if self.well_col not in df.columns:
            raise ValueError(
                f"Column '{self.well_col}' not found in X. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Get unique wells
        unique_wells = df[self.well_col].unique()
        n_wells = len(unique_wells)
        
        if n_wells < self.n_splits:
            raise ValueError(
                f"Number of unique wells ({n_wells}) must be >= n_splits ({self.n_splits})"
            )
        
        # Shuffle wells if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            unique_wells = rng.permutation(unique_wells)
        
        # Split wells into folds
        fold_size = n_wells // self.n_splits
        remainder = n_wells % self.n_splits
        
        start = 0
        for fold in range(self.n_splits):
            # Adjust fold size for remainder
            end = start + fold_size + (1 if fold < remainder else 0)
            
            # Get test wells for this fold
            test_wells = unique_wells[start:end]
            
            # Get indices
            test_mask = df[self.well_col].isin(test_wells)
            train_mask = ~test_mask
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            self.logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"{len(train_indices)} train, {len(test_indices)} test samples, "
                f"{len(test_wells)} test wells"
            )
            
            yield train_indices, test_indices
            
            start = end
    
    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame, optional
            Input data
        y : np.ndarray or pd.Series, optional
            Target values
        groups : np.ndarray, optional
            Group labels
            
        Returns
        -------
        int
            Number of splits
        """
        return self.n_splits


class SpatialCrossValidator(BaseCrossValidator):
    """
    Spatial cross-validation that groups by geographic proximity.
    
    Groups data points by spatial location to prevent spatial correlation
    from causing data leakage between train and test sets.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: Optional[str] = None,
        distance_threshold: Optional[float] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize spatial cross-validator.
        
        Parameters
        ----------
        n_splits : int, default 5
            Number of folds
        x_col : str, default 'X'
            Column name for X coordinates
        y_col : str, default 'Y'
            Column name for Y coordinates
        z_col : str, optional
            Column name for Z coordinates (if 3D)
        distance_threshold : float, optional
            Maximum distance for grouping points (if None, uses clustering)
        shuffle : bool, default False
            Whether to shuffle groups
        random_state : int, optional
            Random seed
        """
        self.n_splits = n_splits
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.distance_threshold = distance_threshold
        self.shuffle = shuffle
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ):
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data with spatial coordinates
        y : np.ndarray or pd.Series, optional
            Target values (unused)
        groups : np.ndarray, optional
            Pre-computed groups (if None, computed from coordinates)
            
        Yields
        ------
        train_indices : np.ndarray
            Training set indices
        test_indices : np.ndarray
            Test set indices
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            raise ValueError(
                "X must be a DataFrame with coordinate columns "
                f"({self.x_col}, {self.y_col})"
            )
        
        df = X.copy()
        
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise ValueError(
                f"Coordinate columns '{self.x_col}' and '{self.y_col}' "
                f"must be present in X"
            )
        
        # Use provided groups or compute from coordinates
        if groups is None:
            # Simple spatial grouping using KMeans clustering
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError(
                    "scikit-learn is required for spatial cross-validation. "
                    "Install with: pip install scikit-learn"
                )
            
            # Prepare coordinates
            coords = df[[self.x_col, self.y_col]].values
            if self.z_col and self.z_col in df.columns:
                coords = np.column_stack([coords, df[self.z_col].values])
            
            # Cluster into n_splits groups
            kmeans = KMeans(
                n_clusters=self.n_splits,
                random_state=self.random_state,
                n_init=10
            )
            groups = kmeans.fit_predict(coords)
        
        # Split by groups
        unique_groups = np.unique(groups)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            unique_groups = rng.permutation(unique_groups)
        
        for fold in range(self.n_splits):
            # Get test group for this fold
            test_group = unique_groups[fold]
            
            # Get indices
            test_mask = groups == test_group
            train_mask = ~test_mask
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            self.logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"{len(train_indices)} train, {len(test_indices)} test samples"
            )
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Returns the number of splitting iterations.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame, optional
            Input data
        y : np.ndarray or pd.Series, optional
            Target values
        groups : np.ndarray, optional
            Group labels
            
        Returns
        -------
        int
            Number of splits
        """
        return self.n_splits
