Petrophysics Guide
==================

Complete guide to petrophysical analysis with GeoSuite.

Overview
--------

The petrophysics module provides tools for:

* Water saturation calculations (Archie equation)
* Porosity and formation factor calculations
* Pickett plot analysis (resistivity-porosity crossplots)
* Buckles plot analysis (bulk volume water)
* Neutron-density crossplots for lithology identification

Archie's Equation
-----------------

Calculate water saturation from resistivity and porosity:

.. code-block:: python

   from geosuite.petro import calculate_water_saturation
   
   sw = calculate_water_saturation(
       resistivity=10.5,  # ohm-m (deep resistivity)
       porosity=0.25,     # fraction (effective porosity)
       rw=0.05,           # ohm-m (formation water resistivity)
       a=1.0,             # tortuosity factor
       m=2.0,             # cementation exponent
       n=2.0              # saturation exponent
   )
   
   print(f"Water saturation: {sw:.1%}")
   print(f"Hydrocarbon saturation: {(1-sw):.1%}")

Pickett Plot
------------

Create resistivity-porosity crossplot:

.. code-block:: python

   from geosuite.petro import pickett_plot
   from geosuite.data import load_demo_well_logs
   
   df = load_demo_well_logs()
   
   fig = pickett_plot(
       df,
       resistivity_col='RESDEEP',
       porosity_col='PHIE',
       show_isolines=True,
       rw=0.05
   )
   fig.savefig('pickett.png', dpi=300)

See :doc:`../api/petro` for complete API reference.

