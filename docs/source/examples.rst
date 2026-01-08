Examples and Tutorials
======================

This section provides comprehensive examples and tutorials for using GeoSuite.

Jupyter Notebooks
-----------------

Complete workflow notebooks are available in the ``examples/notebooks/`` directory:

.. toctree::
   :maxdepth: 2

Existing Workflow Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``examples/notebooks/`` directory contains several comprehensive notebooks demonstrating GeoSuite workflows:

- **las-loader.ipynb**: LAS file loading and processing
- **facies-classification-svm.ipynb**: SVM-based facies classification
- **change-point-analysis.ipynb**: Stratigraphic change-point detection
- **building-a-geomechanical-model-from-lots.ipynb**: Geomechanical modeling
- **pickett-plot.ipynb**: Pickett plot analysis
- **map-wells.ipynb**: Well mapping and visualization

These notebooks provide complete, working examples of GeoSuite functionality.

Additional Notebooks
--------------------

Other example notebooks in ``examples/notebooks/``:

- ``las-loader.ipynb``: LAS file loading and processing
- ``facies-classification-svm.ipynb``: SVM-based facies classification
- ``change-point-analysis.ipynb``: Stratigraphic change-point detection
- ``building-a-geomechanical-model-from-lots.ipynb``: Geomechanical modeling
- ``pickett-plot.ipynb``: Pickett plot analysis
- ``map-wells.ipynb``: Well mapping and visualization

Running Notebooks
-----------------

To run the notebooks:

1. Install GeoSuite with all optional dependencies:

   .. code-block:: bash

      pip install geosuite[all]

2. Start Jupyter:

   .. code-block:: bash

      jupyter notebook

3. Navigate to ``examples/notebooks/`` and open the desired notebook

Script Examples
---------------

Python script examples are available in ``examples/scripts/``:

- ``quickstart_demo.py``: Quick start demonstration
- ``petrophysics_example.py``: Petrophysical calculations
- ``ml_facies_example.py``: Machine learning examples
- ``changepoint_example.py``: Change-point detection

These can be run directly:

.. code-block:: bash

   python examples/scripts/quickstart_demo.py
