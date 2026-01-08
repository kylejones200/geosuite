"""
Hyperparameter optimization for subsurface machine learning models.

Provides Optuna integration and subsurface-specific optimization strategies
for tuning model hyperparameters.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score, make_scorer

logger = logging.getLogger(__name__)

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning(
        "Optuna not available. Hyperparameter optimization requires Optuna. "
        "Install with: pip install optuna"
    )
    optuna = None
    TPESampler = None
    RandomSampler = None
    MedianPruner = None
    SuccessiveHalvingPruner = None


class SubsurfaceHyperparameterOptimizer:
    """
    Hyperparameter optimizer with subsurface-specific strategies.
    
    Provides optimized search spaces and strategies for common subsurface
    ML tasks like facies classification and petrophysical property prediction.
    
    Example:
        >>> from geosuite.ml import SubsurfaceHyperparameterOptimizer
        >>> from geosuite.ml import PermeabilityPredictor
        >>> 
        >>> optimizer = SubsurfaceHyperparameterOptimizer(
        ...     task_type='regression',
        ...     n_trials=50
        ... )
        >>> 
        >>> best_params = optimizer.optimize(
        ...     PermeabilityPredictor,
        ...     X_train, y_train,
        ...     cv=5
        ... )
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        n_trials: int = 50,
        timeout: Optional[float] = None,
        direction: str = 'maximize',
        sampler: str = 'tpe',
        pruner: str = 'median',
        random_state: int = 42
    ):
        """
        Initialize hyperparameter optimizer.
        
        Parameters
        ----------
        task_type : str, default 'classification'
            Type of task: 'classification' or 'regression'
        n_trials : int, default 50
            Number of optimization trials
        timeout : float, optional
            Maximum time in seconds for optimization
        direction : str, default 'maximize'
            Direction of optimization: 'maximize' or 'minimize'
        sampler : str, default 'tpe'
            Sampling strategy: 'tpe' (Tree-structured Parzen Estimator) or 'random'
        pruner : str, default 'median'
            Pruning strategy: 'median' or 'halving'
        random_state : int, default 42
            Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install with: pip install optuna"
            )
        
        self.task_type = task_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.random_state = random_state
        
        sampler_map = {
            'tpe': TPESampler(seed=random_state),
            'random': RandomSampler(seed=random_state)
        }
        self.sampler = sampler_map.get(sampler, TPESampler(seed=random_state))
        
        pruner_map = {
            'median': MedianPruner(),
            'halving': SuccessiveHalvingPruner()
        }
        self.pruner = pruner_map.get(pruner, MedianPruner())
        
        self.study = None
        self.best_params_ = None
        self.best_score_ = None
    
    def _suggest_params(
        self,
        trial: Any,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        model_type : str
            Type of model
            
        Returns
        -------
        dict
            Suggested hyperparameters
        """
        params = {}
        
        if model_type == 'random_forest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500, step=50)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        elif model_type == 'gradient_boosting':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500, step=50)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        elif model_type == 'svm':
            params['C'] = trial.suggest_float('C', 0.1, 100.0, log=True)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            params['kernel'] = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        elif model_type == 'deep':
            params['hidden_layers_1'] = trial.suggest_int('hidden_layers_1', 32, 256)
            params['hidden_layers_2'] = trial.suggest_int('hidden_layers_2', 16, 128)
            params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        return params
    
    def optimize(
        self,
        model_class: type,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_type: Optional[str] = None,
        cv: int = 5,
        scoring: Optional[str] = None,
        **fixed_params
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Parameters
        ----------
        model_class : type
            Model class to optimize (e.g., PermeabilityPredictor)
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets
        model_type : str, optional
            Type of model (auto-detected if not provided)
        cv : int, default 5
            Number of cross-validation folds
        scoring : str, optional
            Scoring metric (auto-selected based on task_type)
        **fixed_params
            Fixed parameters to pass to model
        
        Returns
        -------
        dict
            Best hyperparameters found
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if model_type is None:
            model_type = self._detect_model_type(model_class)
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        def objective(trial):
            params = self._suggest_params(trial, model_type)
            params.update(fixed_params)
            
            model = model_class(**params)
            
            if self.task_type == 'classification':
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            
            scores = cross_val_score(
                model, X, y,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1
            )
            
            return scores.mean()
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value
        
        logger.info(f"Optimization complete. Best score: {self.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params_}")
        
        return self.best_params_
    
    def _detect_model_type(self, model_class: type) -> str:
        """Detect model type from class name."""
        class_name = model_class.__name__.lower()
        
        if 'randomforest' in class_name or 'random_forest' in class_name:
            return 'random_forest'
        elif 'gradient' in class_name or 'boosting' in class_name:
            return 'gradient_boosting'
        elif 'svm' in class_name or 'svc' in class_name:
            return 'svm'
        elif 'deep' in class_name or 'neural' in class_name:
            return 'deep'
        else:
            return 'random_forest'
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        try:
            import optuna.visualization as vis
            fig = vis.plot_optimization_history(self.study)
            return fig
        except ImportError:
            logger.warning("Optuna visualization requires plotly. Install with: pip install plotly")
            return None
    
    def plot_param_importances(self):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        try:
            import optuna.visualization as vis
            fig = vis.plot_param_importances(self.study)
            return fig
        except ImportError:
            logger.warning("Optuna visualization requires plotly. Install with: pip install plotly")
            return None


def optimize_facies_classifier(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model_type: str = 'random_forest',
    n_trials: int = 50,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to optimize facies classifier hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Training features
    y : np.ndarray or pd.Series
        Training targets (facies labels)
    model_type : str, default 'random_forest'
        Type of model to optimize
    n_trials : int, default 50
        Number of optimization trials
    cv : int, default 5
        Number of cross-validation folds
        
    Returns
    -------
    dict
        Best hyperparameters
    """
    from .enhanced_classifiers import MLflowFaciesClassifier
    
    optimizer = SubsurfaceHyperparameterOptimizer(
        task_type='classification',
        n_trials=n_trials
    )
    
    return optimizer.optimize(
        MLflowFaciesClassifier,
        X, y,
        model_type=model_type,
        cv=cv
    )


def optimize_property_predictor(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model_type: str = 'random_forest',
    n_trials: int = 50,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to optimize property predictor hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Training features
    y : np.ndarray or pd.Series
        Training targets (property values)
    model_type : str, default 'random_forest'
        Type of model to optimize
    n_trials : int, default 50
        Number of optimization trials
    cv : int, default 5
        Number of cross-validation folds
        
    Returns
    -------
    dict
        Best hyperparameters
    """
    from .regression import PermeabilityPredictor
    
    optimizer = SubsurfaceHyperparameterOptimizer(
        task_type='regression',
        n_trials=n_trials
    )
    
    return optimizer.optimize(
        PermeabilityPredictor,
        X, y,
        model_type=model_type,
        cv=cv
    )

