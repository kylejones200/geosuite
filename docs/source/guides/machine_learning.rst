Machine Learning Guide
======================

Complete guide to machine learning workflows with GeoSuite.

Overview
--------

The ML module provides tools for:

* Facies classification with multiple algorithms
* MLflow integration for experiment tracking
* Confusion matrix analysis
* Model evaluation and comparison

Facies Classification
---------------------

Train a classifier on well log data:

.. code-block:: python

   from geosuite.ml import train_facies_classifier
   from geosuite.data import load_facies_training_data
   
   # Load Kansas University benchmark dataset
   df = load_facies_training_data()
   
   # Train model
   results = train_facies_classifier(
       df,
       feature_cols=['GR', 'NPHI', 'RHOB', 'PE'],
       target_col='Facies',
       model_type='random_forest',
       n_estimators=200,
       max_depth=20
   )
   
   print(f"Test accuracy: {results['test_accuracy']:.1%}")

See :doc:`../api/ml` for complete API reference.

