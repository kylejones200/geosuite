Utilities Module
================

Helper functions and compatibility layers.

.. automodule:: geosuite.utils
   :members:
   :undoc-members:
   :show-inheritance:

Numba Helpers
-------------

Compatibility layer for Numba JIT compilation with graceful fallback.

.. automodule:: geosuite.utils.numba_helpers
   :members:
   :undoc-members:
   :show-inheritance:

This module provides a compatibility layer that allows GeoSuite to function
with or without Numba installed. If Numba is available, functions are JIT-compiled
for 10-100x speedups. If not, functions fall back to pure Python with no errors.

Example
~~~~~~~

.. code-block:: python

   from geosuite.utils.numba_helpers import njit, prange, NUMBA_AVAILABLE
   
   @njit(cache=True)
   def fast_loop(data):
       result = 0
       for i in prange(len(data)):
           result += data[i]
       return result
   
   # Check if Numba is active
   if NUMBA_AVAILABLE:
       print("Using Numba JIT compilation")
   else:
       print("Using pure Python fallback")

