Stratigraphy Guide
==================

Complete guide to stratigraphic analysis with GeoSuite.

Overview
--------

The stratigraphy module provides tools for:

* Automated change-point detection for formation tops
* Bayesian online detection (Numba-optimized, 70x faster)
* PELT algorithm
* Log preprocessing and denoising

Change-Point Detection
----------------------

Detect formation boundaries from well logs:

.. code-block:: python

   from geosuite.stratigraphy import (
       preprocess_log,
       detect_bayesian_online
   )
   from geosuite.data import load_demo_well_logs
   
   df = load_demo_well_logs()
   
   # Preprocess log
   gr_clean = preprocess_log(df['GR'].values, method='median', window=5)
   
   # Numba-optimized: 70x faster
   cp_indices, cp_probs = detect_bayesian_online(
       gr_clean,
       expected_segment_length=100,
       threshold=0.5
   )
   
   print(f"Detected {len(cp_indices)} formation boundaries")
   print(f"At depths: {df['DEPTH'].iloc[cp_indices].values}")

See :doc:`../api/stratigraphy` for complete API reference.

