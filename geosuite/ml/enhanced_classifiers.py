"""
Enhanced ML Classifiers with MLflow integration for GeoSuite.
Extends existing classifiers with experiment tracking and model registry.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import sys
from typing import Dict, Tuple, List, Any, Optional, Union

from ..base.estimators import BaseEstimator

logger = logging.getLogger(__name__)

# Make MLflow imports optional
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow not available. Install with: pip install mlflow")
    mlflow = None
    MLFLOW_AVAILABLE = False

# Add the MLflow service to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'services'))

try:
    from mlflow_service import MLflowService
except ImportError:
    logger.warning("MLflowService not available, using basic MLflow functionality")
    MLflowService = None


class MLflowFaciesClassifier(BaseEstimator):
    """Enhanced facies classifier with MLflow experiment tracking."""
    
    def __init__(self, mlflow_service: MLflowService = None):
        """
        Initialize the MLflow-enhanced facies classifier.
        
        Args:
            mlflow_service: MLflowService instance for tracking
        """
        super().__init__()
        self.mlflow_service = mlflow_service or (MLflowService() if MLflowService else None)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.classes = None
        self.current_run_id = None
        self._estimator_type = 'classifier'
        
    def prepare_synthetic_data(self, n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic facies data for training and testing.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        np.random.seed(random_state)
        
        # Generate synthetic well log features
        X = pd.DataFrame({
            'GR': np.random.normal(75, 25, n_samples),
            'NPHI': np.random.normal(0.15, 0.05, n_samples),
            'RHOB': np.random.normal(2.5, 0.2, n_samples),
            'PE': np.random.normal(3.0, 0.5, n_samples),
            'DEPTH': np.random.uniform(1000, 3000, n_samples),
            'DTC': np.random.normal(200, 50, n_samples),  # Compressional transit time
            'RILD': np.random.lognormal(1, 1, n_samples)  # Deep resistivity
        })
        
        # Apply constraints to make data more realistic
        X['GR'] = np.clip(X['GR'], 0, 200)
        X['NPHI'] = np.clip(X['NPHI'], 0, 0.5)
        X['RHOB'] = np.clip(X['RHOB'], 1.8, 3.0)
        X['PE'] = np.clip(X['PE'], 0.5, 8.0)
        X['DTC'] = np.clip(X['DTC'], 50, 500)
        X['RILD'] = np.clip(X['RILD'], 0.1, 1000)
        
        # Generate facies labels based on log characteristics
        conditions = [
            (X['GR'] < 40) & (X['NPHI'] < 0.08) & (X['RHOB'] > 2.6),     # Clean Sand
            (X['GR'] < 60) & (X['NPHI'] < 0.15) & (X['RHOB'] > 2.4),     # Shaly Sand
            (X['GR'] >= 60) & (X['GR'] < 90) & (X['NPHI'] < 0.2),        # Siltstone
            (X['GR'] >= 90) & (X['NPHI'] > 0.15),                         # Shale
            (X['RHOB'] > 2.8) & (X['PE'] > 4.5),                         # Carbonate
            (X['NPHI'] > 0.3) & (X['GR'] < 70)                           # Coal/Organic
        ]
        
        choices = ['Clean_Sand', 'Shaly_Sand', 'Siltstone', 'Shale', 'Carbonate', 'Coal']
        y = pd.Series(np.select(conditions, choices, default='Mudstone'))
        
        return X, y
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        scale_features: bool = True,
        cv_folds: int = 5,
        **model_params
    ) -> 'MLflowFaciesClassifier':
        """
        Fit the classifier to training data.
        
        Args:
            X: Feature array or DataFrame
            y: Target array or Series
            model_type: Type of model to train, default 'random_forest'
            test_size: Fraction of data to use for testing, default 0.2
            scale_features: Whether to scale features, default True
            cv_folds: Number of cross-validation folds, default 5
            **model_params: Additional model parameters
            
        Returns:
            self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        self.train_model(
            X, y,
            model_type=model_type,
            test_size=test_size,
            scale_features=scale_features,
            cv_folds=cv_folds,
            **model_params
        )
        return self
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = 'random_forest',
                   test_size: float = 0.2,
                   scale_features: bool = True,
                   cv_folds: int = 5,
                   **model_params) -> Dict[str, Any]:
        """
        Train a facies classification model with MLflow tracking.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: Type of model to train ('random_forest', 'svm', 'gradient_boosting', 'logistic')
            test_size: Fraction of data to use for testing
            scale_features: Whether to scale features
            cv_folds: Number of cross-validation folds
            **model_params: Additional model parameters
            
        Returns:
            Dictionary with training results
        """
        self.feature_names = list(X.columns)
        self.classes = list(y.unique())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            self.scaler = None
        
        # Initialize model based on type
        models = {
            'random_forest': RandomForestClassifier(random_state=42, **model_params),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, **model_params),
            'svm': SVC(random_state=42, probability=True, **model_params),
            'logistic': LogisticRegression(random_state=42, max_iter=1000, **model_params)
        }
        
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = models[model_type]
        
        # Start MLflow run
        if self.mlflow_service:
            run_name = f"facies_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            run = self.mlflow_service.start_run(
                run_name=run_name,
                tags={
                    "model_type": "facies_classification",
                    "algorithm": model_type,
                    "domain": "geology",
                    "scaled_features": str(scale_features)
                }
            )
            self.current_run_id = run.info.run_id
        else:
            run = mlflow.start_run()
            self.current_run_id = run.info.run_id
        
        try:
            # Log parameters
            params = {
                "model_type": model_type,
                "test_size": test_size,
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "n_features": X_train.shape[1],
                "n_classes": len(self.classes),
                "scaled_features": scale_features,
                "cv_folds": cv_folds
            }
            params.update(model_params)
            mlflow.log_params(params)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Log metrics
            metrics = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std
            }
            mlflow.log_metrics(metrics)
            
            # Log additional info
            mlflow.log_param("classes", self.classes)
            mlflow.log_param("features", self.feature_names)
            
            # Generate classification report
            class_report = classification_report(y_test, y_test_pred, output_dict=True)
            
            # Log per-class metrics
            for class_name in self.classes:
                if class_name in class_report:
                    mlflow.log_metric(f"precision_{class_name}", class_report[class_name]['precision'])
                    mlflow.log_metric(f"recall_{class_name}", class_report[class_name]['recall'])
                    mlflow.log_metric(f"f1_score_{class_name}", class_report[class_name]['f1-score'])
            
            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save and log feature importance
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    feature_importance.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, "feature_importance")
                os.unlink(f.name)
                
                # Log top feature importances as metrics
                for i, (feature, importance) in enumerate(feature_importance.head(5).itertuples(index=False)):
                    mlflow.log_metric(f"top_feature_{i+1}_importance", importance)
            
            # Log confusion matrix
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                pd.DataFrame(conf_matrix, index=self.classes, columns=self.classes).to_csv(f.name)
                mlflow.log_artifact(f.name, "confusion_matrix")
            os.unlink(f.name)
            
            # Log model
            if self.mlflow_service:
                # Use MLflow service for enhanced logging
                self.mlflow_service.log_facies_classification_experiment(
                    model=self.model,
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_test=X_test_scaled,
                    y_test=y_test,
                    accuracy=test_accuracy,
                    feature_names=self.feature_names
                )
            else:
                # Basic MLflow logging
                signature = infer_signature(X_train_scaled.values, y_train_pred)
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="facies_model",
                    signature=signature,
                    input_example=X_train_scaled.head(3).values
                )
            
            # Log scaler if used
            if self.scaler:
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
                    import joblib
                    joblib.dump(self.scaler, f.name)
                    mlflow.log_artifact(f.name, "scaler")
                os.unlink(f.name)
            
            results = {
                "run_id": self.current_run_id,
                "model_type": model_type,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
                "feature_importance": feature_importance if hasattr(self.model, 'feature_importances_') else None,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix,
                "classes": self.classes,
                "feature_names": self.feature_names
            }
            
            return results
            
        finally:
            mlflow.end_run()
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature array or DataFrame
            
        Returns:
            Predicted class labels as numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X_processed = X.copy()
        if self.scaler:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        X_processed = X.copy()
        if self.scaler:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        return self.model.predict_proba(X_processed)
    
    def register_model(self, model_name: str, description: str = None) -> str:
        """
        Register the trained model in MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            description: Model description
            
        Returns:
            Model version
        """
        if not self.mlflow_service or not self.current_run_id:
            raise ValueError("MLflow service not available or no active run")
        
        version = self.mlflow_service.register_model(
            run_id=self.current_run_id,
            model_name=model_name,
            model_version_description=description
        )
        
        return version


def train_facies_classifier(model_type: str = 'random_forest',
                           n_samples: int = 1000,
                           test_size: float = 0.2,
                           **model_params) -> Dict[str, Any]:
    """
    Convenience function to train a facies classifier with MLflow tracking.
    
    Args:
        model_type: Type of model to train
        n_samples: Number of synthetic samples to generate
        test_size: Test set size
        **model_params: Model-specific parameters
        
    Returns:
        Training results dictionary
    """
    classifier = MLflowFaciesClassifier()
    
    # Generate synthetic data
    X, y = classifier.prepare_synthetic_data(n_samples=n_samples)
    
    # Train model
    results = classifier.train_model(
        X=X,
        y=y,
        model_type=model_type,
        test_size=test_size,
        **model_params
    )
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    logger.info("Training facies classifier with MLflow tracking...")
    
    # Train a Random Forest model
    results = train_facies_classifier(
        model_type='random_forest',
        n_samples=2000,
        n_estimators=100,
        max_depth=10
    )
    
    logger.info("Training completed!")
    logger.info(f"Run ID: {results['run_id']}")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.3f}")
    logger.info(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}")
