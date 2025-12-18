Installation
============

Requirements
------------

* Python 3.12 or higher
* pip package manager

Basic Installation
------------------

Install GeoSuite with pip:

.. code-block:: bash

   pip install geosuite

This installs the core library with all required dependencies:

* numpy
* pandas
* scipy
* scikit-learn
* matplotlib
* seaborn
* numba (for performance optimization)
* lasio (for LAS files)
* segyio (for SEG-Y files)
* ruptures (for change-point detection)

Optional Dependencies
---------------------

GeoSuite has several optional feature sets that can be installed:

Machine Learning
~~~~~~~~~~~~~~~~

For MLflow integration and experiment tracking:

.. code-block:: bash

   pip install geosuite[ml]

Includes:

* mlflow >= 2.8.0
* Additional ML utilities

Geospatial
~~~~~~~~~~

For spatial analysis with Apache Sedona:

.. code-block:: bash

   pip install geosuite[geospatial]

Includes:

* apache-sedona
* pyspark
* geopandas
* h3

Web Application
~~~~~~~~~~~~~~~

For the Flask/Dash web interface:

.. code-block:: bash

   pip install geosuite[webapp]

Includes:

* Flask
* Dash
* Plotly
* gunicorn

Imaging
~~~~~~~

For core image processing:

.. code-block:: bash

   pip install geosuite[imaging]

Includes:

* scikit-image
* opencv-python

Development
~~~~~~~~~~~

For development and testing:

.. code-block:: bash

   pip install geosuite[dev]

Includes:

* pytest
* black
* ruff
* mypy
* sphinx

All Features
~~~~~~~~~~~~

To install everything:

.. code-block:: bash

   pip install geosuite[all]

From Source
-----------

To install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/geosuite.git
   cd geosuite
   pip install -e .

For development with all optional dependencies:

.. code-block:: bash

   pip install -e ".[dev,all]"

Verification
------------

Verify your installation:

.. code-block:: python

   import geosuite
   print(geosuite.__version__)
   
   # Check Numba availability
   from geosuite.utils.numba_helpers import NUMBA_AVAILABLE
   print(f"Numba available: {NUMBA_AVAILABLE}")

Run the test suite:

.. code-block:: bash

   pytest

Run performance benchmarks:

.. code-block:: bash

   python benchmarks/bench_numba_speedup.py

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors for optional dependencies, install the appropriate extra:

.. code-block:: bash

   pip install geosuite[ml]  # for MLflow
   pip install geosuite[geospatial]  # for Sedona

Numba Issues
~~~~~~~~~~~~

If Numba fails to install or compile:

* Ensure you have Python 3.12 or higher
* Update your compilers (gcc/clang on Linux/Mac, MSVC on Windows)
* GeoSuite will fall back to pure Python if Numba is unavailable

Performance will be reduced without Numba but all functionality will work.

SEG-Y/LAS Issues
~~~~~~~~~~~~~~~~

If you have issues with segyio or lasio:

.. code-block:: bash

   pip install --upgrade segyio lasio

Platform-Specific Notes
-----------------------

macOS
~~~~~

On Apple Silicon (M1/M2), ensure you're using a native ARM64 Python:

.. code-block:: bash

   python --version  # Should show arm64
   pip install geosuite

Windows
~~~~~~~

On Windows, you may need Visual C++ build tools for some dependencies:

* Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
* Install "Desktop development with C++"

Linux
~~~~~

On Linux, ensure development headers are installed:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install python3-dev

   # CentOS/RHEL
   sudo yum install python3-devel

