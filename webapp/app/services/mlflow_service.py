"""
MLflow Service for GeoSuite ML Model Management.
Handles experiment tracking, model registry, and deployment.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import joblib
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowService:
    """Service for managing MLflow experiments, models, and deployments."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "geosuite"):
        """
        Initialize MLflow service.
        
        Args:
            tracking_uri: MLflow tracking server URI. If None, uses local file store.
            experiment_name: Default experiment name for GeoSuite models.
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"MLflow service initialized with tracking URI: {self.tracking_uri}")
        logger.info(f"Using experiment: {self.experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> "mlflow.ActiveRun":
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add to the run
            
        Returns:
            MLflow active run object
        """
        run_tags = {
            "application": "geosuite",
            "timestamp": datetime.now().isoformat()
        }
        if tags:
            run_tags.update(tags)
            
        return mlflow.start_run(run_name=run_name, tags=run_tags)
    
    def log_model_training(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray = None, y_test: np.ndarray = None,
                          model_name: str = "model", params: Dict = None,
                          metrics: Dict = None) -> str:
        """
        Log a trained model with MLflow.
        
        Args:
            model: Trained scikit-learn model
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            model_name: Name for the model
            params: Model parameters to log
            metrics: Model metrics to log
            
        Returns:
            Run ID of the logged model
        """
        with mlflow.start_run() as run:
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=X_train[:5] if len(X_train) > 5 else X_train
            )
            
            # Log training data info
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1] if X_train.ndim > 1 else 1)
            
            if X_test is not None:
                mlflow.log_param("n_test_samples", len(X_test))
                
            # Log model metadata
            mlflow.set_tag("model_type", type(model).__name__)
            mlflow.set_tag("framework", "scikit-learn")
            
            return run.info.run_id
    
    def log_facies_classification_experiment(self, model: Any, X_train: pd.DataFrame,
                                           y_train: pd.Series, X_test: pd.DataFrame = None,
                                           y_test: pd.Series = None, accuracy: float = None,
                                           feature_names: List[str] = None) -> str:
        """
        Log a facies classification experiment.
        
        Args:
            model: Trained classification model
            X_train: Training features DataFrame
            y_train: Training labels Series
            X_test: Test features DataFrame (optional)
            y_test: Test labels Series (optional)
            accuracy: Model accuracy score
            feature_names: List of feature names
            
        Returns:
            Run ID
        """
        with self.start_run(
            run_name=f"facies_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "facies_classification", "domain": "geology"}
        ) as run:
            
            # Log model parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log dataset information
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(y_train.unique()))
            mlflow.log_param("classes", list(y_train.unique()))
            
            if feature_names:
                mlflow.log_param("features", feature_names)
            
            # Log metrics
            if accuracy is not None:
                mlflow.log_metric("accuracy", accuracy)
            
            if X_test is not None and y_test is not None:
                test_accuracy = model.score(X_test, y_test)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_param("n_test_samples", len(X_test))
            
            # Log model
            signature = infer_signature(X_train.values, model.predict(X_train.values))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="facies_model",
                signature=signature,
                input_example=X_train.head(3).values
            )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_names or [f'feature_{i}' for i in range(len(model.feature_importances_))],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save and log feature importance
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    feature_importance.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, "feature_importance")
                os.unlink(f.name)
            
            return run.info.run_id
    
    def register_model(self, run_id: str, model_name: str, model_version_description: str = None) -> str:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name to register the model under
            model_version_description: Description for this model version
            
        Returns:
            Model version number
        """
        try:
            # Get the model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Update model version description
            if model_version_description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=model_version_description
                )
            
            logger.info(f"Model registered: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def load_model_from_registry(self, model_name: str, version: str = "latest", stage: str = None):
        """
        Load a model from MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            version: Model version ("latest" or specific version number)
            stage: Model stage ("Staging", "Production", etc.)
            
        Returns:
            Loaded model
        """
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        elif version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_uri}: {e}")
            raise
    
    def get_registered_models(self) -> List[Dict]:
        """
        Get all registered models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.search_registered_models()
            model_info = []
            
            for model in models:
                latest_version = self.client.get_latest_versions(
                    model.name, stages=["Production", "Staging", "None"]
                )
                
                model_info.append({
                    "name": model.name,
                    "description": model.description,
                    "latest_version": latest_version[0].version if latest_version else "None",
                    "creation_time": model.creation_timestamp,
                    "last_updated": model.last_updated_timestamp
                })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get registered models: {e}")
            return []
    
    def get_experiment_runs(self, experiment_name: str = None, max_results: int = 50) -> pd.DataFrame:
        """
        Get experiment runs as a DataFrame.
        
        Args:
            experiment_name: Experiment name (uses default if None)
            max_results: Maximum number of runs to return
            
        Returns:
            DataFrame with run information
        """
        exp_name = experiment_name or self.experiment_name
        
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if not experiment:
                return pd.DataFrame()
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            runs_data = []
            for run in runs:
                run_data = {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
                    "end_time": pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None
                }
                
                # Add parameters and metrics
                run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
                run_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
                run_data.update({f"tag_{k}": v for k, v in run.data.tags.items()})
                
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            return pd.DataFrame()
    
    def promote_model_to_stage(self, model_name: str, version: str, stage: str) -> bool:
        """
        Promote a model version to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version to promote
            stage: Target stage ("Staging", "Production", "Archived")
            
        Returns:
            Success status
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} promoted to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the registered model
            version: Model version to delete
            
        Returns:
            Success status
        """
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            logger.info(f"Model version {model_name} v{version} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
    
    def get_model_performance_comparison(self, model_names: List[str]) -> pd.DataFrame:
        """
        Compare performance of different registered models.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            DataFrame with model performance comparison
        """
        comparison_data = []
        
        for model_name in model_names:
            try:
                # Get latest version
                latest_versions = self.client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
                
                for version in latest_versions:
                    # Get run metrics
                    run = self.client.get_run(version.run_id)
                    
                    comparison_data.append({
                        "model_name": model_name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "accuracy": run.data.metrics.get("accuracy"),
                        "test_accuracy": run.data.metrics.get("test_accuracy"),
                        "creation_time": pd.to_datetime(version.creation_timestamp, unit='ms')
                    })
                    
            except Exception as e:
                logger.warning(f"Could not get performance data for {model_name}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_old_runs(self, experiment_name: str = None, keep_last_n: int = 50) -> int:
        """
        Clean up old experiment runs, keeping only the most recent ones.
        
        Args:
            experiment_name: Experiment name (uses default if None)
            keep_last_n: Number of recent runs to keep
            
        Returns:
            Number of runs deleted
        """
        exp_name = experiment_name or self.experiment_name
        
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if not experiment:
                return 0
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if len(runs) <= keep_last_n:
                return 0
            
            runs_to_delete = runs[keep_last_n:]
            deleted_count = 0
            
            for run in runs_to_delete:
                try:
                    self.client.delete_run(run.info.run_id)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete run {run.info.run_id}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old runs from experiment {exp_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            return 0
