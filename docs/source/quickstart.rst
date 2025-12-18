Quick Start
===========

This guide will get you up and running with GeoSuite in minutes.

First Steps
-----------

After installing GeoSuite, let's start with a simple example:

.. code-block:: python

   import geosuite
   from geosuite.data import load_demo_well_logs
   
   # Load demo data
   df = load_demo_well_logs()
   print(df.head())

This loads a sample well log dataset with common log curves.

Petrophysics Example
--------------------

Calculate water saturation using Archie's equation:

.. code-block:: python

   from geosuite.petro import calculate_water_saturation
   
   # Calculate water saturation
   sw = calculate_water_saturation(
       resistivity=10.5,  # ohm-m
       porosity=0.25,     # fraction
       rw=0.05,           # water resistivity
       a=1.0,             # tortuosity factor
       m=2.0,             # cementation exponent
       n=2.0              # saturation exponent
   )
   
   print(f"Water Saturation: {sw:.1%}")
   # Output: Water Saturation: 44.7%

Create a Pickett plot:

.. code-block:: python

   from geosuite.petro import pickett_plot
   
   df = load_demo_well_logs()
   fig = pickett_plot(
       df, 
       resistivity_col='RESDEEP',
       porosity_col='PHIE'
   )
   fig.savefig('pickett.png')

Geomechanics Example
---------------------

Calculate overburden stress (Numba-optimized, 25x faster):

.. code-block:: python

   from geosuite.geomech import calculate_overburden_stress
   from geosuite.data import load_demo_well_logs
   
   df = load_demo_well_logs()
   
   # Calculate vertical stress from density log
   sv = calculate_overburden_stress(df['DEPTH'], df['RHOB'])
   
   # Add to dataframe
   df['Sv'] = sv
   
   print(f"Overburden at {df['DEPTH'].iloc[-1]:.0f}m: {sv[-1]:.2f} MPa")

Calculate pore pressure:

.. code-block:: python

   from geosuite.geomech import (
       calculate_pore_pressure_eaton,
       calculate_hydrostatic_pressure
   )
   
   # Hydrostatic pressure
   pp_hydro = calculate_hydrostatic_pressure(df['DEPTH'])
   
   # Eaton method prediction
   pp_eaton = calculate_pore_pressure_eaton(
       depths=df['DEPTH'],
       observed_velocity=df['DTC'],
       overburden=sv
   )
   
   df['PP_Eaton'] = pp_eaton

Stratigraphy Example
--------------------

Detect formation boundaries (Numba-optimized, 70x faster):

.. code-block:: python

   from geosuite.stratigraphy import (
       preprocess_log,
       detect_bayesian_online
   )
   
   # Preprocess gamma ray log
   gr_clean = preprocess_log(
       df['GR'].values,
       method='median',
       window=5
   )
   
   # Detect change points
   cp_indices, cp_probs = detect_bayesian_online(
       gr_clean,
       expected_segment_length=100,
       threshold=0.5
   )
   
   print(f"Detected {len(cp_indices)} formation boundaries")
   print(f"At depths: {df['DEPTH'].iloc[cp_indices].values}")

Machine Learning Example
------------------------

Train a facies classifier:

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
       n_estimators=200
   )
   
   print(f"Test Accuracy: {results['test_accuracy']:.1%}")
   print(f"Model saved to: {results['model_path']}")

Complete Workflow
-----------------

Here's a complete workflow combining multiple modules:

.. code-block:: python

   import pandas as pd
   from geosuite.data import load_demo_well_logs
   from geosuite.geomech import (
       calculate_overburden_stress,
       calculate_pore_pressure_eaton,
       stress_polygon_limits
   )
   from geosuite.stratigraphy import detect_bayesian_online, preprocess_log
   from geosuite.plotting import create_strip_chart
   
   # Load data
   df = load_demo_well_logs()
   
   # Geomechanics calculations (Numba-optimized)
   df['Sv'] = calculate_overburden_stress(df['DEPTH'], df['RHOB'])
   df['PP'] = calculate_pore_pressure_eaton(
       df['DEPTH'], df['DTC'], df['Sv']
   )
   
   # Stratigraphy (Numba-optimized)
   gr_clean = preprocess_log(df['GR'].values)
   cp_indices, _ = detect_bayesian_online(gr_clean)
   
   # Create visualization
   fig = create_strip_chart(
       df,
       depth_col='DEPTH',
       log_cols=['GR', 'RHOB', 'Sv', 'PP'],
       change_points=cp_indices
   )
   fig.savefig('workflow_result.png', dpi=300, bbox_inches='tight')
   
   print(f"Analysis complete!")
   print(f"  Overburden at TD: {df['Sv'].iloc[-1]:.2f} MPa")
   print(f"  Pore pressure at TD: {df['PP'].iloc[-1]:.2f} MPa")
   print(f"  Detected {len(cp_indices)} formation boundaries")

Performance Tips
----------------

GeoSuite uses Numba JIT compilation for performance optimization:

First Run (Cold Start)
~~~~~~~~~~~~~~~~~~~~~~

The first time you call a Numba-optimized function, it will compile:

.. code-block:: python

   import time
   from geosuite.geomech import calculate_overburden_stress
   import numpy as np
   
   depth = np.linspace(0, 3000, 10000)
   rhob = np.ones(10000) * 2.5
   
   # First call - includes compilation time (~1 second)
   start = time.time()
   sv = calculate_overburden_stress(depth, rhob)
   print(f"First call: {time.time() - start:.3f}s")
   
   # Second call - uses cached compiled code (very fast)
   start = time.time()
   sv = calculate_overburden_stress(depth, rhob)
   print(f"Second call: {time.time() - start:.3f}s")
   # Output: Second call: 0.001s (1000x faster!)

Check Numba Status
~~~~~~~~~~~~~~~~~~

Verify Numba is available:

.. code-block:: python

   from geosuite.utils.numba_helpers import NUMBA_AVAILABLE
   
   if NUMBA_AVAILABLE:
       print("[OK] Numba enabled - maximum performance")
   else:
       print("[WARNING] Numba not available - using fallback mode")

Next Steps
----------

* Read the :doc:`guides/petrophysics` for detailed workflows
* Explore the :doc:`api/index` for complete API reference
* Check :doc:`performance` for optimization details
* See :doc:`examples` for more complete examples

Getting Help
------------

* Documentation: https://geosuite.readthedocs.io
* GitHub Issues: https://github.com/yourusername/geosuite/issues
* Examples: See ``examples/`` directory in the repository

