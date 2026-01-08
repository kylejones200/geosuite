"""
Example: Integrating GeoSuite with pygeomodeling for spatial reservoir modeling.

This example demonstrates how to:
1. Load well log data using GeoSuite
2. Calculate petrophysical properties
3. Convert to spatial format
4. Model properties spatially using pygeomodeling
5. Predict on a 3D grid
"""
import numpy as np
import pandas as pd

# Check if pygeomodeling is available
try:
    from geosuite.modeling import (
        WellLogToSpatial,
        SpatialPropertyModeler,
        create_reservoir_model,
        interpolate_properties,
        SpatialDataConverter,
        PYGEO_AVAILABLE
    )
    
    if not PYGEO_AVAILABLE:
        print("pygeomodeling not available. Install with: pip install pygeomodeling")
        exit(1)
except ImportError as e:
    print(f"Modeling module not available: {e}")
    print("Install with: pip install geosuite[modeling]")
    exit(1)

from geosuite.io import load_csv_data
from geosuite.petro import (
    calculate_porosity_from_density,
    calculate_water_saturation,
    calculate_formation_factor,
)


def example_single_well_spatial_modeling():
    """Example: Model a single well's properties spatially."""
    print("=" * 80)
    print("Example 1: Single Well Spatial Modeling")
    print("=" * 80)
    
    # Generate synthetic well log data (in practice, load from LAS/CSV)
    depth = np.linspace(0, 3000, 1000)
    df = pd.DataFrame({
        'DEPTH': depth,
        'GR': np.random.normal(75, 25, 1000),
        'NPHI': np.random.normal(0.15, 0.05, 1000),
        'RHOB': np.random.normal(2.5, 0.2, 1000),
        'RT': np.random.lognormal(2, 1, 1000),
    })
    
    # Calculate petrophysical properties
    df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
    df['WATER_SAT'] = calculate_water_saturation(
        df['POROSITY'], df['RT'], rw=0.05
    )
    df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4) / (df['WATER_SAT'] ** 2)  # Timur equation
    
    # Convert to spatial format (assuming well at coordinates 1000, 2000)
    converter = WellLogToSpatial()
    spatial_data = converter.convert(
        df,
        x=1000.0,
        y=2000.0,
        z='DEPTH',
        properties=['POROSITY', 'PERMEABILITY', 'WATER_SAT'],
        well_name='Well_001'
    )
    
    print(f"\nConverted {len(spatial_data)} depth samples to spatial format")
    print(f"Spatial data columns: {list(spatial_data.columns)}")
    
    # Model permeability spatially
    modeler = SpatialPropertyModeler(model_type='gpr')
    modeler.fit_property(spatial_data, 'PERMEABILITY')
    
    # Create prediction grid
    grid = SpatialDataConverter.create_prediction_grid(
        x_range=(500, 1500),
        y_range=(1500, 2500),
        z_range=(0, 3000),
        nx=20,
        ny=20,
        nz=10
    )
    
    # Predict
    predictions, uncertainty = modeler.predict(grid, return_std=True)
    
    print(f"\nPredicted permeability at {len(grid)} grid points")
    print(f"Mean permeability: {np.mean(predictions):.2f} mD")
    print(f"Mean uncertainty: {np.mean(uncertainty):.2f} mD")
    
    return spatial_data, predictions, uncertainty


def example_multi_well_reservoir_model():
    """Example: Create a reservoir model from multiple wells."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Well Reservoir Model")
    print("=" * 80)
    
    # Generate data for multiple wells
    wells = {}
    coordinates = {}
    
    for i, (x, y) in enumerate([(1000, 2000), (1500, 2500), (2000, 3000)], 1):
        well_name = f'Well_{i:03d}'
        depth = np.linspace(0, 3000, 500)
        
        df = pd.DataFrame({
            'DEPTH': depth,
            'RHOB': np.random.normal(2.5, 0.2, 500),
            'RT': np.random.lognormal(2, 1, 500),
        })
        
        # Calculate properties
        df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
        df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4)
        
        wells[well_name] = df
        coordinates[well_name] = (x, y)
    
    print(f"\nLoaded {len(wells)} wells")
    
    # Create reservoir model
    model = create_reservoir_model(
        well_data=wells,
        coordinates=coordinates,
        properties=['POROSITY', 'PERMEABILITY'],
        model_type='gpr'
    )
    
    # Create prediction grid
    grid = SpatialDataConverter.create_prediction_grid(
        x_range=(0, 3000),
        y_range=(0, 4000),
        z_range=(0, 3000),
        nx=30,
        ny=30,
        nz=15
    )
    
    # Predict all properties
    predictions = model.predict_all_properties(grid, return_std=False)
    
    print(f"\nPredicted {len(predictions)} properties at {len(grid)} grid points")
    for prop_name, pred in predictions.items():
        print(f"  {prop_name}: mean = {np.mean(pred):.4f}, std = {np.std(pred):.4f}")
    
    return model, predictions


def example_workflow_integration():
    """Example: Complete workflow from well logs to spatial model."""
    print("\n" + "=" * 80)
    print("Example 3: Complete Workflow Integration")
    print("=" * 80)
    
    # Step 1: Load well log data (simulated)
    depth = np.linspace(0, 3000, 1000)
    df = pd.DataFrame({
        'DEPTH': depth,
        'GR': np.random.normal(75, 25, 1000),
        'NPHI': np.random.normal(0.15, 0.05, 1000),
        'RHOB': np.random.normal(2.5, 0.2, 1000),
        'RT': np.random.lognormal(2, 1, 1000),
    })
    
    print("\nStep 1: Loaded well log data")
    print(f"  Samples: {len(df)}")
    print(f"  Depth range: {df['DEPTH'].min():.0f} - {df['DEPTH'].max():.0f} m")
    
    # Step 2: Calculate petrophysical properties
    df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
    df['WATER_SAT'] = calculate_water_saturation(df['POROSITY'], df['RT'], rw=0.05)
    df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4) / (df['WATER_SAT'] ** 2)
    
    print("\nStep 2: Calculated petrophysical properties")
    print(f"  Porosity range: {df['POROSITY'].min():.3f} - {df['POROSITY'].max():.3f}")
    print(f"  Permeability range: {df['PERMEABILITY'].min():.2f} - {df['PERMEABILITY'].max():.2f} mD")
    
    # Step 3: Convert to spatial format
    converter = WellLogToSpatial()
    spatial_data = converter.convert(
        df,
        x=1000.0,
        y=2000.0,
        z='DEPTH',
        properties=['POROSITY', 'PERMEABILITY'],
    )
    
    print("\nStep 3: Converted to spatial format")
    print(f"  Spatial samples: {len(spatial_data)}")
    
    # Step 4: Interpolate to grid
    grid = SpatialDataConverter.create_prediction_grid(
        x_range=(500, 1500),
        y_range=(1500, 2500),
        z_range=(0, 3000),
        nx=25,
        ny=25,
        nz=12
    )
    
    results, uncertainty = interpolate_properties(
        spatial_data,
        'PERMEABILITY',
        grid,
        return_uncertainty=True
    )
    
    print("\nStep 4: Interpolated to 3D grid")
    print(f"  Grid points: {len(grid)}")
    print(f"  Mean permeability: {results['PERMEABILITY'].mean():.2f} mD")
    print(f"  Mean uncertainty: {uncertainty['PERMEABILITY_STD'].mean():.2f} mD")
    
    return results, uncertainty


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("GeoSuite + pygeomodeling Integration Examples")
    print("=" * 80)
    print("\nThis example demonstrates integration between GeoSuite well log")
    print("analysis and pygeomodeling spatial reservoir modeling.\n")
    
    try:
        # Run examples
        example_single_well_spatial_modeling()
        example_multi_well_reservoir_model()
        example_workflow_integration()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

