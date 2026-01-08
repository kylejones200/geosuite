"""
Model interpretability tools for machine learning models.

Provides feature importance, SHAP values, and partial dependence plots
for understanding model predictions.
"""
import logging
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def get_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Supports scikit-learn models with feature_importances_ attribute
    (Random Forest, Gradient Boosting, etc.).
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list, optional
        Names of features. If None, uses indices.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores
        
    Raises
    ------
    AttributeError
        If model doesn't have feature_importances_ attribute
    """
    # Try to get feature importances from model or pipeline
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        steps = (
            model.named_steps if hasattr(model, 'named_steps') else
            dict(model.steps) if hasattr(model, 'steps') else
            None
        )
        
        if steps is None:
            raise AttributeError(
                "Model does not have feature_importances_ attribute. "
                "Supported models: RandomForest, GradientBoosting, etc."
            )
        
        importances = next(
            (step_model.feature_importances_ for step_model in steps.values() 
             if hasattr(step_model, 'feature_importances_')),
            None
        )
        
        if importances is None:
            raise AttributeError(
                "Model does not have feature_importances_ attribute. "
                "Supported models: RandomForest, GradientBoosting, etc."
            )
    
    # Get feature names
    if feature_names is None:
        feature_names = (
            list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else
            [f"Feature_{i}" for i in range(model.n_features_in_)] if hasattr(model, 'n_features_in_') else
            [f"Feature_{i}" for i in range(len(importances))]
        )
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Extracted feature importance for {len(importance_df)} features")
    
    return importance_df


def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (8, 6)
) -> Figure:
    """
    Plot feature importance from a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_
    feature_names : list, optional
        Names of features
    top_n : int, optional
        Number of top features to show (if None, shows all)
    figsize : tuple, default (8, 6)
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    try:
        import signalplot
        signalplot.apply()
    except ImportError:
        pass
    
    importance_df = get_feature_importance(model, feature_names)
    
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(
        range(len(importance_df)),
        importance_df['importance'].values,
        align='center'
    )
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Feature Importance', fontsize=12, pad=10)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig


def calculate_shap_values(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    max_samples: Optional[int] = 100,
    feature_names: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Calculate SHAP values for model interpretability.
    
    Requires shap library to be installed.
    
    Parameters
    ----------
    model : Any
        Trained model
    X : np.ndarray or pd.DataFrame
        Input features
    max_samples : int, optional
        Maximum number of samples to use (for large datasets)
        
    Returns
    -------
    np.ndarray or None
        SHAP values array, or None if shap is not available
        
    Raises
    ------
    ImportError
        If shap library is not installed
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap library is required for SHAP values. "
            "Install with: pip install shap"
        )
    
    # Limit samples if needed
    if max_samples is not None and len(X) > max_samples:
        X_sample = (
            X.sample(n=max_samples, random_state=42) if isinstance(X, pd.DataFrame) else
            X[np.random.choice(len(X), max_samples, replace=False)]
        )
        logger.info(f"Using {max_samples} samples for SHAP calculation")
    else:
        X_sample = X
    
    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class for binary/multi-class
        
        logger.info(f"Calculated SHAP values: shape {shap_values.shape}")
        return shap_values
        
    except Exception as e:
        logger.warning(f"TreeExplainer failed, trying KernelExplainer: {e}")
        try:
            explainer = shap.KernelExplainer(model.predict, X_sample[:10])
            shap_values = explainer.shap_values(X_sample)
            return shap_values
        except Exception as e2:
            logger.error(f"Failed to calculate SHAP values: {e2}")
            return None


def plot_shap_summary(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    max_samples: Optional[int] = 100,
    figsize: tuple = (10, 8)
) -> Optional[Figure]:
    """
    Create SHAP summary plot.
    
    Parameters
    ----------
    model : Any
        Trained model
    X : np.ndarray or pd.DataFrame
        Input features
    feature_names : list, optional
        Feature names
    max_samples : int, optional
        Maximum samples for SHAP calculation
    figsize : tuple, default (10, 8)
        Figure size
        
    Returns
    -------
    Figure or None
        Matplotlib figure, or None if shap is not available
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap library not available, skipping SHAP plot")
        return None
    
    shap_values = calculate_shap_values(model, X, max_samples=max_samples)
    
    if shap_values is None:
        return None
    
    # Limit X for plotting
    if max_samples is not None and len(X) > max_samples:
        X_plot = (
            X.sample(n=max_samples, random_state=42) if isinstance(X, pd.DataFrame) else
            X[np.random.choice(len(X), max_samples, replace=False)]
        )
    else:
        X_plot = X
    
    # Create SHAP summary plot
    fig = plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_plot, feature_names=feature_names, show=False)
    
    return fig


def partial_dependence_plot(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature: Union[int, str],
    feature_names: Optional[List[str]] = None,
    n_points: int = 50,
    figsize: tuple = (8, 6)
) -> Figure:
    """
    Create partial dependence plot for a single feature.
    
    Parameters
    ----------
    model : Any
        Trained model
    X : np.ndarray or pd.DataFrame
        Training data
    feature : int or str
        Feature index or name
    feature_names : list, optional
        Feature names (if X is array)
    n_points : int, default 50
        Number of points to evaluate
    figsize : tuple, default (8, 6)
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    try:
        from sklearn.inspection import partial_dependence
    except ImportError:
        raise ImportError(
            "scikit-learn is required for partial dependence plots. "
            "Install with: pip install scikit-learn"
        )
    
    try:
        import signalplot
        signalplot.apply()
    except ImportError:
        pass
    
    # Get feature index
    if isinstance(feature, str):
        feature_idx = (
            X.columns.get_loc(feature) if isinstance(X, pd.DataFrame) else
            feature_names.index(feature) if feature_names else
            (lambda: (_ for _ in ()).throw(ValueError(f"Feature '{feature}' not found")))()
        )
    else:
        feature_idx = feature
    
    # Calculate partial dependence
    pd_result = partial_dependence(
        model, X, features=[feature_idx], grid_resolution=n_points
    )
    
    # Extract values
    feature_values = pd_result['grid_values'][0]
    pd_values = pd_result['average'][0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(feature_values, pd_values, linewidth=2)
    ax.set_xlabel(
        feature if isinstance(feature, str) else f"Feature {feature}",
        fontsize=11
    )
    ax.set_ylabel('Partial Dependence', fontsize=11)
    ax.set_title('Partial Dependence Plot', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

