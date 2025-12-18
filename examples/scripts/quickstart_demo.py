#!/usr/bin/env python3
"""
GeoSuite Quick Start Demo
==========================

This script demonstrates the basic usage of GeoSuite library.

Run with:
    python quickstart_demo.py
"""

import logging
import geosuite
from geosuite import data, petro, geomech, ml
import pandas as pd

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    logger.info("GeoSuite Quick Start Demo")
    
    # 1. Show version and available modules
    logger.info("1. GeoSuite Information")
    info = geosuite.get_info()
    logger.info(f"Version: {info['version']}")
    logger.info(f"Available modules: {', '.join([k for k, v in info['modules'].items() if v])}")
    
    # 2. List available demo datasets
    logger.info("\n2. Available Demo Datasets")
    datasets = geosuite.list_demo_datasets()
    
    # 3. Load demo well log data
    logger.info("\n3. Loading Demo Well Log Data")
    df_logs = data.load_demo_well_logs()
    logger.info(f"Loaded {len(df_logs)} rows")
    logger.info(f"Columns: {', '.join(df_logs.columns)}")
    logger.info("\nFirst few rows:")
    logger.info(f"\n{df_logs.head()}")
    
    # 4. Petrophysics - Calculate water saturation
    logger.info("\n4. Petrophysics Example - Water Saturation")
    sw = petro.calculate_water_saturation(
        resistivity=10.5,
        porosity=0.25,
        a=1.0,
        m=2.0,
        n=2.0,
        rw=0.05
    )
    logger.info(f"Input: Resistivity=10.5 ohm-m, Porosity=25%")
    logger.info(f"Output: Water Saturation = {sw:.2%}")
    
    # 5. Geomechanics - Calculate overburden stress
    logger.info("\n5. Geomechanics Example - Overburden Stress")
    depths = [1000, 1500, 2000, 2500, 3000]  # meters
    densities = [2.3, 2.35, 2.4, 2.45, 2.5]  # g/cc
    sv = geomech.calculate_overburden_stress(depths, densities)
    
    logger.info("Depth (m) | Density (g/cc) | Overburden (MPa)")
    for d, rho, s in zip(depths, densities, sv):
        logger.info(f"{d:8.0f} | {rho:13.2f} | {s:16.2f}")
    
    # 6. Machine Learning - Load facies data
    logger.info("\n6. Machine Learning Example - Facies Data")
    df_facies = data.load_facies_training_data()
    logger.info(f"Loaded {len(df_facies)} rows of facies training data")
    logger.info(f"Features: {', '.join([c for c in df_facies.columns if c not in ['Facies', 'Formation', 'Well Name']])}")
    logger.info(f"Number of wells: {df_facies['Well Name'].nunique()}")
    logger.info("Facies distribution:")
    logger.info(f"\n{df_facies['Facies'].value_counts().sort_index()}")
    
    # 7. Summary
    logger.info("\nDemo completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  - Explore example notebooks in examples/notebooks/")
    logger.info("  - Check out the full documentation in docs/")
    logger.info("  - Try the web application: cd webapp && python app.py")
    logger.info("  - Read the README.md for more examples")


if __name__ == "__main__":
    main()

