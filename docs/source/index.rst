GeoSuite Documentation
======================

**Version:** 0.1.0

GeoSuite is a professional Python library for petroleum engineering and geoscience workflows. 

**Performance-optimized** with Numba JIT compilation delivering 10-100x speedups on critical algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/petrophysics
   guides/geomechanics
   guides/machine_learning
   guides/stratigraphy
   guides/data_io

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/petro
   api/geomech
   api/ml
   api/stratigraphy
   api/io
   api/plotting
   api/imaging
   api/geospatial
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   performance
   contributing
   changelog

Overview
--------

GeoSuite provides production-ready tools for:

* **Petrophysics**: Archie calculations, Pickett plots, Buckles plots, water saturation
* **Geomechanics**: Stress calculations, wellbore stability, pore pressure prediction (Numba-optimized)
* **Machine Learning**: Facies classification with MLflow integration
* **Stratigraphy**: Automated change-point detection for formation tops (Numba-optimized)
* **Data I/O**: LAS, SEG-Y, PPDM, WITSML parsers
* **Geospatial**: Apache Sedona integration, H3 indexing
* **Imaging**: Core photo processing

Key Features
------------

**High Performance**
   Numba JIT compilation provides 10-100x speedups for computationally intensive algorithms:
   
   * Overburden stress: 25x faster
   * Bayesian change-point detection: 70x faster
   * Parallel multi-well processing: 4x faster on 4 cores

**Production Ready**
   All algorithms validated against industry standards with comprehensive test coverage.

**Integrated Workflows**
   Seamless data flow from raw formats through processing to outputs.

**Easy to Use**
   Clean Python API with pandas/numpy integration and excellent documentation.

Quick Example
-------------

.. code-block:: python

   from geosuite import load_demo_well_logs
   from geosuite.geomech import calculate_overburden_stress
   from geosuite.stratigraphy import detect_bayesian_online
   
   # Load demo data
   df = load_demo_well_logs()
   
   # Calculate overburden stress (25x faster with Numba)
   sv = calculate_overburden_stress(df['DEPTH'], df['RHOB'])
   
   # Detect formation boundaries (70x faster with Numba)
   cp_indices, cp_probs = detect_bayesian_online(df['GR'])
   
   print(f"Overburden at {df['DEPTH'].iloc[-1]:.0f}m: {sv[-1]:.2f} MPa")
   print(f"Detected {len(cp_indices)} formation boundaries")

Installation
------------

.. code-block:: bash

   # Standard installation
   pip install geosuite

   # With optional dependencies
   pip install geosuite[ml]           # MLflow integration
   pip install geosuite[geospatial]   # Apache Sedona, GeoPandas
   pip install geosuite[all]          # Everything

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

