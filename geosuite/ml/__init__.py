from .classifiers import train_and_predict
from .confusion_matrix_utils import (
    display_cm,
    display_adj_cm,
    confusion_matrix_to_dataframe,
    compute_metrics_from_cm,
    plot_confusion_matrix
)
from .regression import PermeabilityPredictor, PorosityPredictor
from .cross_validation import WellBasedKFold, SpatialCrossValidator
from .interpretability import (
    get_feature_importance,
    plot_feature_importance,
    calculate_shap_values,
    plot_shap_summary,
    partial_dependence_plot,
)
from .clustering import (
    FaciesClusterer,
    cluster_facies,
    find_optimal_clusters,
)

__all__ = [
    "train_and_predict",
    "display_cm",
    "display_adj_cm",
    "confusion_matrix_to_dataframe",
    "compute_metrics_from_cm",
    "plot_confusion_matrix",
    "PermeabilityPredictor",
    "PorosityPredictor",
    "WellBasedKFold",
    "SpatialCrossValidator",
    "get_feature_importance",
    "plot_feature_importance",
    "calculate_shap_values",
    "plot_shap_summary",
    "partial_dependence_plot",
    "FaciesClusterer",
    "cluster_facies",
    "find_optimal_clusters",
]

# Make MLflow-enhanced classifiers optional
try:
    from .enhanced_classifiers import MLflowFaciesClassifier, train_facies_classifier
    __all__.extend(["MLflowFaciesClassifier", "train_facies_classifier"])
except ImportError:
    pass
