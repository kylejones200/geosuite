Data I/O Guide
==============

Complete guide to loading and exporting data with GeoSuite.

Overview
--------

The I/O module provides parsers for:

* LAS files (Log ASCII Standard)
* SEG-Y files (seismic data)
* PPDM CSV files (petroleum data model)
* WITSML XML (wellsite information transfer)

Loading LAS Files
-----------------

.. code-block:: python

   from geosuite.io import load_las_file
   
   # Load LAS file
   las = load_las_file('path/to/file.las')
   
   # Convert to DataFrame
   df = las.df()

Loading WITSML Files
--------------------

.. code-block:: python

   from geosuite.io import WitsmlParser
   
   parser = WitsmlParser()
   wells = parser.parse_file('path/to/file.xml')
   
   for well in wells:
       print(f"Well: {well.name}")
       print(f"Logs: {len(well.logs)}")

See :doc:`../api/io` for complete API reference.

