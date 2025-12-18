#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)
"""
GeoSuite Machine Learning Example - Facies Classification
==========================================================

This script demonstrates facies classification using machine learning with GeoSuite.

Run with:
    python ml_facies_example.py
"""

from geosuite.data import load_facies_training_data, load_kansas_training_wells
from geosuite.ml import train_facies_classifier, compute_metrics_from_cm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    logger.info("GeoSuite Machine Learning Example - Facies Classification")

    
    # Load facies training data
    logger.info("1. Loading Facies Training Data")
    
    df = load_facies_training_data()
    logger.info(f"Loaded {len(df)} rows from {df['Well Name'].nunique()} wells")
    logger.info(f"\nColumns: {', '.join(df.columns)}")
    logger.info(f"\nFacies distribution:")
    logger.info(df['Facies'].value_counts().sort_index())
    
    
    # Define features and target
    logger.info("2. Defining Features and Target")
    
    feature_cols = ['GR', 'NPHI', 'RHOB', 'PE', 'DEPTH']
    target_col = 'Facies'
    
    # Check if all features are available
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {', '.join(available_features)}")
    logger.info(f"Target variable: {target_col}")
    
    
    if len(available_features) < 3:
        logger.info("ERROR: Not enough features available for training.")
        logger.info("Available columns:", df.columns.tolist())
        return
    
    # Prepare data
    X = df[available_features].fillna(df[available_features].median())
    y = df[target_col]
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    
    # Train multiple models
    models = ['random_forest', 'svm', 'gradient_boosting', 'logistic_regression']
    results_list = []
    
    logger.info("3. Training Multiple Models")
    logger.info("-" * 70)
    
    for model_type in models:
        logger.info(f"\nTraining {model_type.replace('_', ' ').title()}...")
        
        try:
            results = train_facies_classifier(
                df=df,
                feature_cols=available_features,
                target_col=target_col,
                model_type=model_type,
                test_size=0.3,
                random_state=42
            )
            
            results_list.append({
                'Model': model_type.replace('_', ' ').title(),
                'Train Accuracy': results['train_accuracy'],
                'Test Accuracy': results['test_accuracy'],
            })
            
            logger.info(f"  Train Accuracy: {results['train_accuracy']:.4f}")
            logger.info(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
            
        except Exception as e:
            logger.info(f"  ERROR: Could not train {model_type}: {e}")
    
    logger.info()
    
    # Compare models
    logger.info("4. Model Comparison")
    
    results_df = pd.DataFrame(results_list)
    logger.info(results_df.to_string(index=False))
    
    
    best_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
    best_accuracy = results_df['Test Accuracy'].max()
    logger.info(f"Best performing model: {best_model} (Test Accuracy: {best_accuracy:.4f})")
    
    
    # Detailed analysis of best model
    logger.info("5. Detailed Analysis - Random Forest")
    
    
    try:
        results = train_facies_classifier(
            df=df,
            feature_cols=available_features,
            target_col=target_col,
            model_type='random_forest',
            test_size=0.3,
            random_state=42,
            n_estimators=100,
            max_depth=10
        )
        
        # Confusion matrix metrics
        logger.info("\nConfusion Matrix:")
        logger.info(results['confusion_matrix'])
        
        
        # Compute per-class metrics
        try:
            metrics_df = compute_metrics_from_cm(
                cm=results['confusion_matrix'],
                labels=results['classes']
            )
            logger.info("Per-class Metrics:")
            logger.info(metrics_df.to_string(index=False))
        except Exception as e:
            logger.info(f"Could not compute detailed metrics: {e}")
        
        
        
    except Exception as e:
        logger.info(f"Could not perform detailed analysis: {e}")
        
    
    # Feature importance (if available)
    logger.info("6. Feature Importance (Random Forest)")
    
    
    try:
        if hasattr(results['model'], 'feature_importances_'):
            importances = results['model'].feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': available_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            logger.info(feature_importance.to_string(index=False))
            
        else:
            logger.info("Feature importance not available for this model.")
            
    except Exception as e:
        logger.info(f"Could not compute feature importance: {e}")
        
    
    # Cross-validation recommendation
    logger.info("7. Next Steps")
    
    logger.info("For production use, consider:")
    logger.info("  - Cross-validation for robust performance estimates")
    logger.info("  - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)")
    logger.info("  - Feature engineering (well-based features, geological context)")
    logger.info("  - Handling class imbalance (SMOTE, class weights)")
    logger.info("  - MLflow for experiment tracking (see ml_mlflow_example.py)")
    logger.info("  - Model interpretation (SHAP values, partial dependence plots)")

    logger.info("Machine Learning example completed!")
    


if __name__ == "__main__":
    main()


