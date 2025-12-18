Machine Learning Module
=======================

Facies classification, confusion matrix utilities, and MLflow integration.

.. automodule:: geosuite.ml
   :members:
   :undoc-members:
   :show-inheritance:

Classifiers
-----------

.. automodule:: geosuite.ml.classifiers
   :members:
   :undoc-members:
   :show-inheritance:

Enhanced Classifiers (MLflow)
------------------------------

.. automodule:: geosuite.ml.enhanced_classifiers
   :members:
   :undoc-members:
   :show-inheritance:

Confusion Matrix Utilities
---------------------------

**Performance Note**: Adjacent facies computation is Numba-optimized for 10-15x speedup.

.. automodule:: geosuite.ml.confusion_matrix_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: geosuite.ml.confusion_matrix_utils.display_cm

.. autofunction:: geosuite.ml.confusion_matrix_utils.display_adj_cm

   **Numba-optimized**: 10-15x faster for adjacent facies computation

