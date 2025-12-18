Stratigraphy Module
===================

Automated change-point detection for formation tops identification.

**Performance Note**: Bayesian detection is Numba-optimized for 70x speedup.

.. automodule:: geosuite.stratigraphy
   :members:
   :undoc-members:
   :show-inheritance:

Change-Point Detection
----------------------

.. automodule:: geosuite.stratigraphy.changepoint
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: geosuite.stratigraphy.preprocess_log

.. autofunction:: geosuite.stratigraphy.detect_pelt

.. autofunction:: geosuite.stratigraphy.detect_bayesian_online

   **Numba-optimized**: 70x faster than pure Python
   
   This function uses JIT compilation for the nested loop structure,
   providing dramatic speedups for large datasets.

.. autofunction:: geosuite.stratigraphy.compare_methods

.. autofunction:: geosuite.stratigraphy.find_consensus

