"""
ML blueprint routes for MLflow model management and experiments.
"""

from flask import render_template, request, jsonify, current_app, redirect, url_for, flash
from . import bp
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

# Add the geosuite_lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'geosuite_lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from services.mlflow_service import MLflowService
    from ml.classifiers import train_facies_classifier
except ImportError:
    # Fallback if modules not available
    MLflowService = None
    train_facies_classifier = None


@bp.route("/")
def ml_home():
    """ML dashboard home page."""
    return render_template("ml/index.html")


@bp.route("/experiments")
def experiments():
    """Experiments tracking page."""
    return render_template("ml/experiments.html")


@bp.route("/models")
def models():
    """Model registry page."""
    return render_template("ml/models.html")


@bp.route("/api/experiments")
def get_experiments():
    """Get experiment runs data."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        mlflow_service = MLflowService()
        runs_df = mlflow_service.get_experiment_runs()
        
        if runs_df.empty:
            return jsonify({
                "status": "success",
                "experiments": [],
                "total": 0
            })
        
        # Convert DataFrame to records
        experiments = runs_df.to_dict('records')
        
        # Format datetime columns for JSON serialization
        for exp in experiments:
            if 'start_time' in exp and exp['start_time']:
                exp['start_time'] = exp['start_time'].isoformat()
            if 'end_time' in exp and exp['end_time']:
                exp['end_time'] = exp['end_time'].isoformat()
        
        return jsonify({
            "status": "success",
            "experiments": experiments,
            "total": len(experiments)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to get experiments",
            "error": str(e)
        }), 500


@bp.route("/api/models")
def get_registered_models():
    """Get registered models from MLflow registry."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        mlflow_service = MLflowService()
        models = mlflow_service.get_registered_models()
        
        return jsonify({
            "status": "success",
            "models": models,
            "total": len(models)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to get registered models",
            "error": str(e)
        }), 500


@bp.route("/api/train-facies-model", methods=["POST"])
def train_facies_model():
    """Train a new facies classification model with MLflow tracking."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        data = request.json
        
        # Get training parameters
        n_estimators = int(data.get('n_estimators', 100))
        max_depth = data.get('max_depth')
        if max_depth:
            max_depth = int(max_depth)
        random_state = int(data.get('random_state', 42))
        test_size = float(data.get('test_size', 0.2))
        
        # Create synthetic training data (in real app, this would come from uploaded data)
        np.random.seed(random_state)
        n_samples = 1000
        
        # Generate synthetic well log features
        X = pd.DataFrame({
            'GR': np.random.normal(75, 25, n_samples),
            'NPHI': np.random.normal(0.15, 0.05, n_samples),
            'RHOB': np.random.normal(2.5, 0.2, n_samples),
            'PE': np.random.normal(3.0, 0.5, n_samples),
            'DEPTH': np.random.uniform(1000, 3000, n_samples)
        })
        
        # Generate synthetic facies labels (simplified classification)
        conditions = [
            (X['GR'] < 50) & (X['NPHI'] < 0.1),  # Clean Sand
            (X['GR'] < 50) & (X['NPHI'] >= 0.1),  # Shaly Sand
            (X['GR'] >= 50) & (X['GR'] < 100),    # Siltstone
            X['GR'] >= 100                        # Shale
        ]
        choices = ['Clean_Sand', 'Shaly_Sand', 'Siltstone', 'Shale']
        y = pd.Series(np.select(conditions, choices, default='Unknown'))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Log with MLflow
        mlflow_service = MLflowService()
        run_id = mlflow_service.log_facies_classification_experiment(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            accuracy=test_accuracy,
            feature_names=list(X.columns)
        )
        
        # Generate classification report
        y_pred = model.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        return jsonify({
            "status": "success",
            "run_id": run_id,
            "model_type": "RandomForestClassifier",
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "classes": list(y.unique()),
            "parameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            },
            "classification_report": class_report
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Model training failed",
            "error": str(e)
        }), 500


@bp.route("/api/register-model", methods=["POST"])
def register_model():
    """Register a model in MLflow Model Registry."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        data = request.json
        run_id = data.get('run_id')
        model_name = data.get('model_name', 'geosuite-facies-classifier')
        description = data.get('description', 'GeoSuite facies classification model')
        
        if not run_id:
            return jsonify({
                "status": "error",
                "message": "run_id is required"
            }), 400
        
        mlflow_service = MLflowService()
        version = mlflow_service.register_model(
            run_id=run_id,
            model_name=model_name,
            model_version_description=description
        )
        
        return jsonify({
            "status": "success",
            "model_name": model_name,
            "version": version,
            "run_id": run_id
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Model registration failed",
            "error": str(e)
        }), 500


@bp.route("/api/promote-model", methods=["POST"])
def promote_model():
    """Promote a model to a different stage."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        data = request.json
        model_name = data.get('model_name')
        version = data.get('version')
        stage = data.get('stage')
        
        if not all([model_name, version, stage]):
            return jsonify({
                "status": "error",
                "message": "model_name, version, and stage are required"
            }), 400
        
        mlflow_service = MLflowService()
        success = mlflow_service.promote_model_to_stage(
            model_name=model_name,
            version=version,
            stage=stage
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Model {model_name} v{version} promoted to {stage}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to promote model"
            }), 500
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Model promotion failed",
            "error": str(e)
        }), 500


@bp.route("/api/load-model", methods=["POST"])
def load_model_for_prediction():
    """Load a model from registry for prediction."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        data = request.json
        model_name = data.get('model_name')
        version = data.get('version', 'latest')
        stage = data.get('stage')
        
        # Sample input data for testing
        sample_data = data.get('sample_data', {
            'GR': 60.0,
            'NPHI': 0.12,
            'RHOB': 2.45,
            'PE': 2.8,
            'DEPTH': 2000.0
        })
        
        if not model_name:
            return jsonify({
                "status": "error",
                "message": "model_name is required"
            }), 400
        
        mlflow_service = MLflowService()
        model = mlflow_service.load_model_from_registry(
            model_name=model_name,
            version=version,
            stage=stage
        )
        
        # Make prediction with sample data
        X_sample = pd.DataFrame([sample_data])
        prediction = model.predict(X_sample)[0]
        prediction_proba = model.predict_proba(X_sample)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "prediction": prediction,
            "sample_input": sample_data
        }
        
        if prediction_proba is not None:
            result["prediction_probabilities"] = {
                class_name: float(prob)
                for class_name, prob in zip(model.classes_, prediction_proba)
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Model loading failed",
            "error": str(e)
        }), 500


@bp.route("/api/compare-models")
def compare_models():
    """Compare performance of registered models."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        model_names = request.args.getlist('models')
        if not model_names:
            # Get all registered models
            mlflow_service = MLflowService()
            registered_models = mlflow_service.get_registered_models()
            model_names = [m['name'] for m in registered_models]
        
        if not model_names:
            return jsonify({
                "status": "success",
                "comparison": [],
                "message": "No models to compare"
            })
        
        mlflow_service = MLflowService()
        comparison_df = mlflow_service.get_model_performance_comparison(model_names)
        
        if comparison_df.empty:
            return jsonify({
                "status": "success",
                "comparison": [],
                "message": "No performance data available"
            })
        
        # Convert DataFrame to records for JSON response
        comparison = comparison_df.to_dict('records')
        
        # Format datetime columns
        for record in comparison:
            if 'creation_time' in record and record['creation_time']:
                record['creation_time'] = record['creation_time'].isoformat()
        
        return jsonify({
            "status": "success",
            "comparison": comparison,
            "total": len(comparison)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Model comparison failed",
            "error": str(e)
        }), 500


@bp.route("/api/cleanup-experiments", methods=["POST"])
def cleanup_experiments():
    """Clean up old experiment runs."""
    try:
        if not MLflowService:
            return jsonify({
                "status": "error",
                "message": "MLflow service not available"
            }), 500
        
        data = request.json
        keep_last_n = int(data.get('keep_last_n', 50))
        
        mlflow_service = MLflowService()
        deleted_count = mlflow_service.cleanup_old_runs(keep_last_n=keep_last_n)
        
        return jsonify({
            "status": "success",
            "deleted_runs": deleted_count,
            "message": f"Cleaned up {deleted_count} old runs"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Cleanup failed",
            "error": str(e)
        }), 500
