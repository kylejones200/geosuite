Geomechanics Module
===================

Stress calculations, pore pressure prediction, and wellbore stability analysis.

**Performance Note**: Core functions are Numba-optimized for 2-25x speedups.

.. automodule:: geosuite.geomech
   :members:
   :undoc-members:
   :show-inheritance:

Pressure Calculations
---------------------

.. automodule:: geosuite.geomech.pressures
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: geosuite.geomech.calculate_overburden_stress

   **Numba-optimized**: 25x faster than pure Python

.. autofunction:: geosuite.geomech.calculate_hydrostatic_pressure

.. autofunction:: geosuite.geomech.calculate_pore_pressure_eaton

.. autofunction:: geosuite.geomech.calculate_pore_pressure_bowers

Stress Calculations
-------------------

.. automodule:: geosuite.geomech.stresses
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: geosuite.geomech.calculate_effective_stress

.. autofunction:: geosuite.geomech.calculate_pressure_gradient

   **Numba-optimized**: 2-5x faster than pure Python

.. autofunction:: geosuite.geomech.calculate_stress_ratio

Stress Polygon
--------------

.. automodule:: geosuite.geomech.stress_polygon
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Processing
-------------------

**Numba parallel processing**: 4x speedup on 4 cores for multi-well workflows.

.. automodule:: geosuite.geomech.parallel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: geosuite.geomech.parallel.calculate_overburden_stress_parallel

.. autofunction:: geosuite.geomech.parallel.get_parallel_info

