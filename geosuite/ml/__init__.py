from .classifiers import train_and_predict
from .confusion_matrix_utils import (
    display_cm,
    display_adj_cm,
    confusion_matrix_to_dataframe,
    compute_metrics_from_cm,
    plot_confusion_matrix
)

__all__ = [
    "train_and_predict",
    "display_cm",
    "display_adj_cm",
    "confusion_matrix_to_dataframe",
    "compute_metrics_from_cm",
    "plot_confusion_matrix",
]

# Make MLflow-enhanced classifiers optional
try:
    from .enhanced_classifiers import MLflowFaciesClassifier, train_facies_classifier
    __all__.extend(["MLflowFaciesClassifier", "train_facies_classifier"])
except ImportError:
    pass
