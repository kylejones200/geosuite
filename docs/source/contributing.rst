Contributing
============

We welcome contributions to GeoSuite! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Run tests
6. Submit a pull request

Development Setup
-----------------

Clone and Install
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/kylejones200/geosuite.git
   cd geosuite
   pip install -e ".[dev,all]"

This installs GeoSuite in editable mode with all development dependencies.

Code Quality Tools
------------------

Formatting
~~~~~~~~~~

We use Black for code formatting:

.. code-block:: bash

   black geosuite/ tests/

Configuration: ``line-length=100`` in ``pyproject.toml``

Linting
~~~~~~~

We use Ruff for linting:

.. code-block:: bash

   ruff check geosuite/ tests/

Fix issues automatically:

.. code-block:: bash

   ruff check --fix geosuite/ tests/

Type Checking
~~~~~~~~~~~~~

We use mypy for type checking:

.. code-block:: bash

   mypy geosuite/

Testing
-------

Run Tests
~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=geosuite --cov-report=html

   # Run specific test file
   pytest tests/unit/test_petro.py

   # Run verbose
   pytest -v

Write Tests
~~~~~~~~~~~

All new features should include tests. Place tests in appropriate locations:

- ``tests/unit/`` - Unit tests for individual functions
- ``tests/integration/`` - Integration tests for workflows
- ``tests/test_numba_accuracy.py`` - Numba optimization tests

Example test:

.. code-block:: python

   import numpy as np
   import pytest
   from geosuite.petro import calculate_water_saturation

   def test_water_saturation_basic():
       """Test basic water saturation calculation."""
       sw = calculate_water_saturation(
           resistivity=10.0,
           porosity=0.25,
           rw=0.05
       )
       assert 0 <= sw <= 1
       assert isinstance(sw, float)

   def test_water_saturation_high_resistivity():
       """High resistivity should give low water saturation."""
       sw = calculate_water_saturation(
           resistivity=1000.0,
           porosity=0.25,
           rw=0.05
       )
       assert sw < 0.1  # Low water saturation in hydrocarbon zone

Documentation
-------------

Docstrings
~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def my_function(x: float, y: float) -> float:
       """
       Brief description of function.
       
       Longer description with more details about what the
       function does and how it works.
       
       Parameters
       ----------
       x : float
           Description of x parameter
       y : float
           Description of y parameter
           
       Returns
       -------
       float
           Description of return value
           
       Examples
       --------
       >>> my_function(1.0, 2.0)
       3.0
       
       Notes
       -----
       Any additional notes or warnings.
       """
       return x + y

Build Documentation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make html

   # Open in browser
   open build/html/index.html

Adding Numba Optimization
--------------------------

When optimizing functions with Numba:

Pattern to Follow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geosuite.utils.numba_helpers import njit
   import numpy as np

   @njit(cache=True)
   def _my_kernel(data):
       """
       Numba-compiled kernel - pure numerical logic.
       
       No pandas, no complex objects, just numpy arrays.
       """
       result = np.zeros(len(data))
       for i in range(len(data)):
           result[i] = data[i] ** 2
       return result

   def my_function(data):
       """
       Public API function.
       
       Parameters
       ----------
       data : array-like
           Input data (can be pandas Series, list, numpy array)
           
       Returns
       -------
       np.ndarray
           Processed result
       """
       # Convert to numpy
       data_arr = np.asarray(data, dtype=np.float64)
       
       # Call optimized kernel
       result = _my_kernel(data_arr)
       
       return result

Testing Optimizations
~~~~~~~~~~~~~~~~~~~~~

Add accuracy tests to ``tests/test_numba_accuracy.py``:

.. code-block:: python

   class TestMyFunctionAccuracy:
       def test_correctness(self):
           """Test numerical correctness."""
           data = np.array([1.0, 2.0, 3.0])
           result = my_function(data)
           expected = np.array([1.0, 4.0, 9.0])
           np.testing.assert_allclose(result, expected)
           
       def test_pandas_compatibility(self):
           """Test with pandas Series."""
           import pandas as pd
           data = pd.Series([1.0, 2.0, 3.0])
           result = my_function(data)
           assert isinstance(result, np.ndarray)

Benchmarking
~~~~~~~~~~~~

Add benchmarks to ``benchmarks/bench_numba_speedup.py``:

.. code-block:: python

   def bench_my_function():
       data = np.random.randn(10000)
       
       # Warmup
       _ = my_function(data[:100])
       
       # Benchmark
       start = time.perf_counter()
       for _ in range(100):
           result = my_function(data)
       elapsed = time.perf_counter() - start
       
       print(f"my_function: {elapsed/100*1000:.3f} ms/run")

Pull Request Process
--------------------

1. **Create a branch**:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. **Make changes**:
   
   - Write code
   - Add tests
   - Update documentation
   - Format code with black
   - Lint with ruff

3. **Commit**:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description"

4. **Push**:

   .. code-block:: bash

      git push origin feature/my-new-feature

5. **Create Pull Request**:
   
   - Go to GitHub
   - Create PR from your branch to main
   - Describe your changes
   - Reference any related issues

PR Checklist
~~~~~~~~~~~~

- [ ] Tests pass (``pytest``)
- [ ] Code formatted (``black``)
- [ ] Linting passes (``ruff``)
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or clearly documented)

Code Review
-----------

All PRs require review before merging. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- API design

Be responsive to feedback and make requested changes.

Reporting Issues
----------------

Use GitHub Issues for:

- Bug reports
- Feature requests
- Documentation improvements
- Performance issues

Bug Report Template
~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   **Describe the bug**
   A clear description of what the bug is.

   **To Reproduce**
   Steps to reproduce the behavior:
   1. Import '...'
   2. Call function '...'
   3. See error

   **Expected behavior**
   What you expected to happen.

   **Actual behavior**
   What actually happened.

   **Environment**
   - OS: [e.g. macOS 12.0]
   - Python version: [e.g. 3.12]
   - GeoSuite version: [e.g. 0.1.0]
   - Numba available: [Yes/No]

   **Additional context**
   Any other context about the problem.

Community Guidelines
--------------------

- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Follow the code of conduct

License
-------

By contributing, you agree that your contributions will be licensed under the same license as GeoSuite.

Questions?
----------

- GitHub Discussions: https://github.com/kylejones200/geosuite/discussions
- GitHub Issues: https://github.com/kylejones200/geosuite/issues

