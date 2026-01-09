Type Hints Guide
================

GeoSuite uses comprehensive type hints throughout the codebase to improve code clarity, enable better IDE support, and facilitate static type checking.

Type Hint Standards
-------------------

GeoSuite follows Python type hinting conventions with some domain-specific patterns:

Basic Types
-----------

.. code-block:: python

   from typing import Union, Optional, List, Dict, Tuple
   import numpy as np
   import pandas as pd

   # Scalar inputs
   def calculate_property(value: float, threshold: int = 10) -> float:
       """Function with scalar types."""
       return value * threshold

   # Optional parameters
   def process_data(data: np.ndarray, filter: Optional[str] = None) -> np.ndarray:
       """Function with optional parameter."""
       if filter:
           # Apply filter
           pass
       return data

Array Types
-----------

GeoSuite functions accept multiple array-like types for flexibility:

.. code-block:: python

   from typing import Union
   import numpy as np
   import pandas as pd

   def calculate_porosity(
       density: Union[np.ndarray, pd.Series, float],
       rho_matrix: float = 2.65
   ) -> np.ndarray:
       """
       Calculate porosity from density.
       
       Parameters
       ----------
       density : np.ndarray, pd.Series, or float
           Bulk density values (g/cc)
       rho_matrix : float, default 2.65
           Matrix density (g/cc)
           
       Returns
       -------
       np.ndarray
           Porosity values (fraction, 0-1)
       """
       # Convert to numpy array
       density_arr = np.asarray(density, dtype=float)
       
       # Calculation
       porosity = (rho_matrix - density_arr) / (rho_matrix - 1.0)
       
       return porosity

Common Patterns
---------------

1. **Array Inputs**: Always use ``Union[np.ndarray, pd.Series]`` for array-like inputs

   .. code-block:: python

      def process_log(log_values: Union[np.ndarray, pd.Series]) -> np.ndarray:
          """Process well log values."""
          return np.asarray(log_values)

2. **Optional Parameters**: Use ``Optional[Type]`` with ``None`` as default

   .. code-block:: python

      def analyze_well(
          df: pd.DataFrame,
          depth_col: Optional[str] = None
      ) -> Dict[str, Any]:
          """Analyze well data."""
          if depth_col is None:
              depth_col = 'DEPTH'
          # ...

3. **Return Types**: Calculation functions return ``np.ndarray``

   .. code-block:: python

      def calculate_stress(depth: np.ndarray) -> np.ndarray:
          """Calculate stress."""
          return depth * 0.0226  # MPa/m

4. **DataFrame Functions**: Return ``pd.DataFrame``

   .. code-block:: python

      def load_well_data(path: str) -> pd.DataFrame:
          """Load well data."""
          return pd.read_csv(path)

5. **Multiple Return Values**: Use ``Tuple``

   .. code-block:: python

      def fit_model(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
          """Fit model and return predictions and score."""
          predictions = model.predict(X)
          score = model.score(X, y)
          return predictions, score

Type Checking
-------------

GeoSuite supports type checking with ``mypy``:

.. code-block:: bash

   # Install mypy
   pip install mypy

   # Type check GeoSuite
   mypy geosuite/

   # Type check specific module
   mypy geosuite/petro/

Type Checking in CI
--------------------

GeoSuite includes ``mypy`` in the CI pipeline with lenient settings to avoid blocking builds while still providing type checking feedback.

Configuration is in ``mypy.ini``:

.. code-block:: ini

   [mypy]
   python_version = 3.12
   warn_return_any = true
   ignore_missing_imports = true
   # ... (see mypy.ini for full configuration)

Examples
--------

Example 1: Petrophysical Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Union
   import numpy as np

   def calculate_water_saturation(
       phi: Union[np.ndarray, pd.Series],
       rt: Union[np.ndarray, pd.Series],
       rw: float = 0.05,
       m: float = 2.0,
       n: float = 2.0
   ) -> np.ndarray:
       """
       Calculate water saturation using Archie's equation.
       
       Type hints indicate:
       - phi and rt can be arrays or Series
       - rw, m, n are scalar floats with defaults
       - Returns numpy array
       """
       phi_arr = np.asarray(phi, dtype=float)
       rt_arr = np.asarray(rt, dtype=float)
       
       F = 1.0 / (phi_arr ** m)
       R0 = F * rw
       sw = (R0 / rt_arr) ** (1.0 / n)
       
       return sw

Example 2: Geomechanical Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Optional, Union
   import numpy as np
   import pandas as pd

   def calculate_pore_pressure(
       depth: Union[np.ndarray, pd.Series],
       velocity: Union[np.ndarray, pd.Series],
       method: str = 'eaton',
       exponent: Optional[float] = None
   ) -> np.ndarray:
       """
       Calculate pore pressure.
       
       Type hints show:
       - Multiple input types accepted
       - Optional parameter with None default
       - Consistent return type
       """
       depth_arr = np.asarray(depth, dtype=float)
       vel_arr = np.asarray(velocity, dtype=float)
       
       if method == 'eaton':
           if exponent is None:
               exponent = 3.0
           # Calculation...
       
       return pressure

Example 3: Machine Learning Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import List, Dict, Any, Union
   import numpy as np
   import pandas as pd

   def train_classifier(
       X: Union[np.ndarray, pd.DataFrame],
       y: Union[np.ndarray, pd.Series],
       model_type: str = 'random_forest',
       **kwargs: Any
   ) -> Dict[str, Any]:
       """
       Train a classifier.
       
       Type hints show:
       - Flexible input types (array or DataFrame)
       - Keyword arguments with Any type
       - Dictionary return type
       """
       # Training logic...
       return {
           'model': trained_model,
           'accuracy': accuracy_score,
           'predictions': y_pred
       }

Best Practices
-------------

1. **Always convert inputs**: Use ``np.asarray()`` to handle multiple input types

2. **Consistent return types**: Calculation functions return ``np.ndarray``

3. **Optional with None**: Use ``Optional[Type] = None`` for optional parameters

4. **Type unions for flexibility**: Use ``Union[np.ndarray, pd.Series]`` for array inputs

5. **Document complex types**: Use docstrings to explain type expectations

6. **Type checking**: Run ``mypy`` regularly to catch type errors

Additional Resources
---------------------

- `Python Type Hints Documentation <https://docs.python.org/3/library/typing.html>`_
- `mypy Documentation <https://mypy.readthedocs.io/>`_
- `PEP 484 - Type Hints <https://www.python.org/dev/peps/pep-0484/>`_


