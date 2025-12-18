Examples
========

Complete working examples demonstrating GeoSuite capabilities.

Example Scripts
---------------

The ``examples/scripts/`` directory contains standalone Python scripts:

Quickstart Demo
~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/scripts/quickstart_demo.py

Demonstrates basic functionality across all modules.

Petrophysics Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/scripts/petrophysics_example.py

Shows Archie calculations, Pickett plots, and Buckles plots.

Machine Learning Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/scripts/ml_facies_example.py

Trains a facies classifier on the Kansas University benchmark dataset.

Change-Point Detection Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/scripts/changepoint_example.py

Demonstrates stratigraphic boundary detection.

Jupyter Notebooks
-----------------

The ``examples/notebooks/`` directory contains 15 interactive notebooks:

1. **Introduction to GeoSuite** - Overview and basic usage
2. **Petrophysics Workflow** - Complete petrophysical analysis
3. **Geomechanics Analysis** - Stress and pressure predictions
4. **Facies Classification** - ML-based facies prediction
5. **Stratigraphy Detection** - Formation boundary identification
6. **Data Loading** - Working with LAS, SEG-Y, PPDM files
7. **Pickett Plot Analysis** - Resistivity-porosity crossplots
8. **Buckles Plot Analysis** - Bulk volume water analysis
9. **Wellbore Stability** - Mud weight window calculations
10. **Pore Pressure Prediction** - Eaton and Bowers methods
11. **MLflow Integration** - Experiment tracking
12. **Multi-Well Analysis** - Batch processing workflows
13. **Visualization** - Creating publication-quality plots
14. **Apache Sedona** - Geospatial analysis
15. **Performance Optimization** - Numba benchmarking

Complete Workflow Example
--------------------------

Here's a complete end-to-end workflow:

.. code-block:: python

   """
   Complete GeoSuite Workflow
   
   This example demonstrates:
   1. Loading well data
   2. Geomechanics calculations (Numba-optimized)
   3. Stratigraphy detection (Numba-optimized)
   4. Petrophysics analysis
   5. Visualization
   """
   
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   from geosuite.data import load_demo_well_logs
   from geosuite.geomech import (
       calculate_overburden_stress,
       calculate_hydrostatic_pressure,
       calculate_pore_pressure_eaton,
       stress_polygon_limits
   )
   from geosuite.stratigraphy import (
       preprocess_log,
       detect_bayesian_online
   )
   from geosuite.petro import (
       calculate_water_saturation,
       pickett_plot
   )
   from geosuite.plotting import create_strip_chart
   
   # Step 1: Load Data
   print("Loading well data...")
   df = load_demo_well_logs()
   print(f"Loaded {len(df)} samples from {df['DEPTH'].min():.0f}m to {df['DEPTH'].max():.0f}m")
   
   # Step 2: Geomechanics Analysis (Numba-optimized)
   print("\nCalculating stresses and pressures...")
   
   # Overburden stress (25x faster with Numba)
   df['Sv'] = calculate_overburden_stress(df['DEPTH'], df['RHOB'])
   
   # Hydrostatic pressure
   df['PP_hydro'] = calculate_hydrostatic_pressure(df['DEPTH'])
   
   # Pore pressure prediction (Eaton method)
   if 'DTC' in df.columns:
       df['PP_eaton'] = calculate_pore_pressure_eaton(
           df['DEPTH'],
           df['DTC'],
           df['Sv']
       )
   
   print(f"  Overburden at TD: {df['Sv'].iloc[-1]:.2f} MPa")
   print(f"  Hydrostatic at TD: {df['PP_hydro'].iloc[-1]:.2f} MPa")
   
   # Step 3: Stratigraphy Detection (Numba-optimized, 70x faster)
   print("\nDetecting formation boundaries...")
   
   # Preprocess gamma ray log
   gr_clean = preprocess_log(
       df['GR'].values,
       method='median',
       window=5
   )
   
   # Detect change points
   cp_indices, cp_probs = detect_bayesian_online(
       gr_clean,
       expected_segment_length=100,
       threshold=0.5
   )
   
   print(f"  Detected {len(cp_indices)} formation boundaries")
   
   # Add to dataframe
   df['Formation_Boundary'] = False
   df.loc[cp_indices, 'Formation_Boundary'] = True
   
   # Step 4: Petrophysics Analysis
   print("\nCalculating water saturation...")
   
   if 'RESDEEP' in df.columns and 'PHIE' in df.columns:
       # Calculate water saturation for each sample
       sw_values = []
       for _, row in df.iterrows():
           if pd.notna(row['RESDEEP']) and pd.notna(row['PHIE']) and row['PHIE'] > 0:
               sw = calculate_water_saturation(
                   resistivity=row['RESDEEP'],
                   porosity=row['PHIE'],
                   rw=0.05
               )
               sw_values.append(sw)
           else:
               sw_values.append(np.nan)
       
       df['Sw'] = sw_values
       
       # Calculate hydrocarbon saturation
       df['Shc'] = 1 - df['Sw']
       
       mean_sw = df['Sw'].mean()
       mean_shc = df['Shc'].mean()
       print(f"  Average water saturation: {mean_sw:.1%}")
       print(f"  Average hydrocarbon saturation: {mean_shc:.1%}")
   
   # Step 5: Identify Reservoir Zones
   print("\nIdentifying reservoir zones...")
   
   # Simple criteria
   if all(col in df.columns for col in ['PHIE', 'Sw', 'GR']):
       df['Reservoir'] = (
           (df['PHIE'] > 0.15) &  # Porosity > 15%
           (df['Sw'] < 0.7) &      # Water saturation < 70%
           (df['GR'] < 70)         # Clean sand (GR < 70 API)
       )
       
       reservoir_thickness = df[df['Reservoir']]['DEPTH'].diff().sum()
       print(f"  Net reservoir thickness: {reservoir_thickness:.1f}m")
   
   # Step 6: Geomechanics - Drilling Window
   print("\nCalculating drilling window...")
   
   # Pick a depth for analysis
   analysis_depth_idx = len(df) // 2
   sv_at_depth = df['Sv'].iloc[analysis_depth_idx]
   pp_at_depth = df['PP_hydro'].iloc[analysis_depth_idx]
   
   # Calculate stress limits
   limits = stress_polygon_limits(
       sv=sv_at_depth,
       pp=pp_at_depth,
       mu=0.6  # Friction coefficient
   )
   
   print(f"  At {df['DEPTH'].iloc[analysis_depth_idx]:.0f}m:")
   print(f"    Min horizontal stress: {limits['shmin_min']:.2f} MPa")
   print(f"    Max horizontal stress: {limits['shmax_max']:.2f} MPa")
   
   # Step 7: Visualization
   print("\nCreating visualizations...")
   
   # Create strip chart
   fig = create_strip_chart(
       df,
       depth_col='DEPTH',
       log_cols=['GR', 'RHOB', 'PHIE', 'RESDEEP', 'Sv', 'PP_hydro'],
       change_points=cp_indices
   )
   fig.savefig('workflow_strip_chart.png', dpi=300, bbox_inches='tight')
   print("  Saved: workflow_strip_chart.png")
   
   # Create Pickett plot (if available)
   if 'RESDEEP' in df.columns and 'PHIE' in df.columns:
       fig_pickett = pickett_plot(
           df,
           resistivity_col='RESDEEP',
           porosity_col='PHIE'
       )
       fig_pickett.savefig('workflow_pickett.png', dpi=300, bbox_inches='tight')
       print("  Saved: workflow_pickett.png")
   
   # Step 8: Export Results
   print("\nExporting results...")
   
   # Save processed data
   output_file = 'workflow_results.csv'
   df.to_csv(output_file, index=False)
   print(f"  Saved: {output_file}")
   
   # Create summary report
   summary = {
       'Well Name': 'Demo Well',
       'TD (m)': df['DEPTH'].max(),
       'Sv at TD (MPa)': df['Sv'].iloc[-1],
       'PP at TD (MPa)': df['PP_hydro'].iloc[-1],
       'Formations Detected': len(cp_indices),
       'Avg Porosity': df['PHIE'].mean() if 'PHIE' in df.columns else None,
       'Avg Water Sat': df['Sw'].mean() if 'Sw' in df.columns else None,
       'Net Reservoir (m)': reservoir_thickness if 'Reservoir' in df.columns else None
   }
   
   summary_df = pd.DataFrame([summary])
   summary_df.to_csv('workflow_summary.csv', index=False)
   print("  Saved: workflow_summary.csv")
   
   print("\n" + "="*80)
   print("WORKFLOW COMPLETE!")
   print("="*80)
   print("\nGenerated files:")
   print("  - workflow_strip_chart.png")
   print("  - workflow_pickett.png")
   print("  - workflow_results.csv")
   print("  - workflow_summary.csv")
   print("\nPerformance notes:")
   print("  Overburden calculation: 25x faster with Numba")
   print("  Change-point detection: 70x faster with Numba")

Performance Example
-------------------

Benchmark Numba optimizations:

.. code-block:: python

   import time
   import numpy as np
   from geosuite.geomech import calculate_overburden_stress
   from geosuite.stratigraphy import detect_bayesian_online
   from geosuite.utils.numba_helpers import NUMBA_AVAILABLE
   
   print(f"Numba available: {NUMBA_AVAILABLE}")
   print("\nPerformance Benchmarks")
   print("="*60)
   
   # Benchmark 1: Overburden Stress
   depth = np.linspace(0, 5000, 10000)
   rhob = np.random.uniform(2.2, 2.7, 10000)
   
   # Warmup
   _ = calculate_overburden_stress(depth[:100], rhob[:100])
   
   # Benchmark
   start = time.perf_counter()
   for _ in range(100):
       sv = calculate_overburden_stress(depth, rhob)
   elapsed = time.perf_counter() - start
   
   print(f"\nOverburden Stress (10K samples, 100 runs):")
   print(f"  Total time: {elapsed:.3f}s")
   print(f"  Per run: {elapsed/100*1000:.3f} ms")
   print(f"  Throughput: {10000*100/elapsed:,.0f} samples/sec")
   
   # Benchmark 2: Bayesian Detection
   signal = np.random.normal(60, 10, 2000)
   
   # Warmup
   _ = detect_bayesian_online(signal[:100])
   
   # Benchmark
   start = time.perf_counter()
   for _ in range(10):
       cp, probs = detect_bayesian_online(signal)
   elapsed = time.perf_counter() - start
   
   print(f"\nBayesian Detection (2K samples, 10 runs):")
   print(f"  Total time: {elapsed:.3f}s")
   print(f"  Per run: {elapsed/10*1000:.1f} ms")
   print(f"  Detected: {len(cp)} change points")

Additional Examples
-------------------

See the following files for more examples:

- ``examples/scripts/`` - Standalone Python scripts
- ``examples/notebooks/`` - Interactive Jupyter notebooks
- ``tests/`` - Unit and integration tests demonstrating API usage
- ``benchmarks/bench_numba_speedup.py`` - Performance benchmarking

