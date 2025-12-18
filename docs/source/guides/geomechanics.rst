Geomechanics Guide
==================

Complete guide to geomechanics analysis with GeoSuite.

Overview
--------

The geomechanics module provides tools for:

* Overburden stress calculation (Numba-optimized, 25x faster)
* Pore pressure prediction (Eaton, Bowers methods)
* Stress polygon analysis
* Wellbore stability assessment
* Mud weight window calculations

Overburden Stress
-----------------

Calculate vertical stress from density log:

.. code-block:: python

   from geosuite.geomech import calculate_overburden_stress
   from geosuite.data import load_demo_well_logs
   
   df = load_demo_well_logs()
   
   # Numba-optimized: 25x faster
   sv = calculate_overburden_stress(df['DEPTH'], df['RHOB'])
   
   df['Sv'] = sv
   print(f"Overburden at TD: {sv[-1]:.2f} MPa")

Pore Pressure Prediction
-------------------------

Using the Eaton method:

.. code-block:: python

   from geosuite.geomech import calculate_pore_pressure_eaton
   
   pp = calculate_pore_pressure_eaton(
       depths=df['DEPTH'],
       observed_velocity=df['DTC'],
       overburden=sv,
       normal_velocity=None,  # Auto-calculate trend
       eaton_exponent=3.0
   )
   
   df['PP'] = pp

See :doc:`../api/geomech` for complete API reference.

