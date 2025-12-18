#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)
"""
GeoSuite Petrophysics Example
==============================

This script demonstrates petrophysics calculations and plots using GeoSuite.

Run with:
    python petrophysics_example.py
"""

from geosuite.data import load_demo_well_logs
from geosuite.petro import (
    calculate_water_saturation,
    calculate_porosity_from_density,
    pickett_plot,
    buckles_plot,
)
import pandas as pd
import numpy as np


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    logger.info("GeoSuite Petrophysics Example")
    
    logger.info()
    
    # Load demo data
    logger.info("Loading demo well log data...")
    df = load_demo_well_logs()
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {', '.join(df.columns)}")
    logger.info()
    
    # Example 1: Calculate water saturation for multiple depths
    logger.info("1. Water Saturation Calculation")
    
    
    # Assuming we have resistivity and porosity in the dataframe
    # If not, let's create synthetic data
    if 'RESDEEP' not in df.columns:
        df['RESDEEP'] = np.random.uniform(1, 100, len(df))
    if 'PHIE' not in df.columns:
        df['PHIE'] = np.random.uniform(0.05, 0.35, len(df))
    
    # Calculate water saturation
    df['SW'] = df.apply(
        lambda row: calculate_water_saturation(
            resistivity=row['RESDEEP'],
            porosity=row['PHIE'],
            a=1.0,
            m=2.0,
            n=2.0,
            rw=0.05
        ),
        axis=1
    )
    
    logger.info("Statistics for calculated water saturation:")
    logger.info(df['SW'].describe())
    logger.info()
    
    # Example 2: Calculate porosity from density
    logger.info("2. Density Porosity Calculation")
    
    
    if 'RHOB' not in df.columns:
        df['RHOB'] = np.random.uniform(2.0, 2.7, len(df))
    
    df['PHI_DENSITY'] = df['RHOB'].apply(
        lambda rhob: calculate_porosity_from_density(
            rhob=rhob,
            rho_matrix=2.65,  # sandstone
            rho_fluid=1.0     # water
        )
    )
    
    logger.info("Statistics for density porosity:")
    logger.info(df['PHI_DENSITY'].describe())
    
    
    # Example 3: Calculate bulk volume water (BVW)
    logger.info("3. Bulk Volume Water (BVW)")
    
    
    df['BVW'] = df['PHIE'] * df['SW']
    
    logger.info("Statistics for BVW:")
    logger.info(df['BVW'].describe())
    
    
    # Identify reservoir quality
    reservoir_quality = pd.cut(
        df['BVW'],
        bins=[0, 0.03, 0.05, 1.0],
        labels=['Good', 'Fair', 'Poor']
    )
    
    logger.info("Reservoir quality distribution:")
    logger.info(reservoir_quality.value_counts())
    
    
    # Example 4: Create Pickett plot
    logger.info("4. Creating Pickett Plot")
    
    logger.info("Generating interactive Pickett plot...")
    
    try:
        fig = pickett_plot(
            df,
            resistivity_col='RESDEEP',
            porosity_col='PHIE',
            title='Pickett Plot - Demo Well'
        )
        
        # Save to HTML
        output_file = 'pickett_plot.html'
        fig.write_html(output_file)
        logger.info(f"Pickett plot saved to: {output_file}")
        logger.info("Open this file in a web browser to view the interactive plot.")
    except Exception as e:
        logger.info(f"Could not create Pickett plot: {e}")
    logger.info()
    
    # Example 5: Create Buckles plot
    logger.info("5. Creating Buckles Plot")
    
    logger.info("Generating interactive Buckles plot...")
    
    try:
        fig = buckles_plot(
            df,
            porosity_col='PHIE',
            sw_col='SW',
            title='Buckles Plot - Demo Well'
        )
        
        # Save to HTML
        output_file = 'buckles_plot.html'
        fig.write_html(output_file)
        logger.info(f"Buckles plot saved to: {output_file}")
        logger.info("Open this file in a web browser to view the interactive plot.")
    except Exception as e:
        logger.info(f"Could not create Buckles plot: {e}")
    logger.info()
    
    # Summary statistics
    logger.info("6. Summary Statistics")
    
    
    summary = pd.DataFrame({
        'Parameter': ['Resistivity (ohm-m)', 'Porosity (fraction)', 'Water Sat (fraction)', 'BVW (fraction)'],
        'Mean': [df['RESDEEP'].mean(), df['PHIE'].mean(), df['SW'].mean(), df['BVW'].mean()],
        'Std': [df['RESDEEP'].std(), df['PHIE'].std(), df['SW'].std(), df['BVW'].std()],
        'Min': [df['RESDEEP'].min(), df['PHIE'].min(), df['SW'].min(), df['BVW'].min()],
        'Max': [df['RESDEEP'].max(), df['PHIE'].max(), df['SW'].max(), df['BVW'].max()],
    })
    
    logger.info(summary.to_string(index=False))
    logger.info()
    
    
    logger.info("Petrophysics example completed!")
    


if __name__ == "__main__":
    main()


