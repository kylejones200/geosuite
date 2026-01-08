# GeoSuite Spatial Modeling Module

Integration with [pygeomodeling](https://github.com/kylejones200/pygeomodeling) for spatial reservoir property modeling using Gaussian Process Regression and Kriging.

## Overview

This module bridges GeoSuite's well log analysis capabilities with pygeomodeling's spatial modeling tools, enabling you to:

- Convert well log data (1D depth-based) to 3D spatial coordinates
- Model reservoir properties (permeability, porosity, etc.) spatially using Gaussian Processes
- Interpolate properties onto 3D grids for reservoir visualization
- Build complete multi-well reservoir models

## Installation

Install with the modeling extra:

```bash
pip install geosuite[modeling]
```

Or install pygeomodeling separately:

```bash
pip install pygeomodeling
```

## Quick Start

### Single Well Spatial Modeling

```python
from geosuite.io import load_las_file
from geosuite.petro import calculate_porosity_from_density
from geosuite.modeling import WellLogToSpatial, SpatialPropertyModeler, SpatialDataConverter

# Load well log data
las = load_las_file('well_001.las')
df = las.df()

# Calculate petrophysical properties
df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4)  # Timur equation

# Convert to spatial format (well at coordinates 1000, 2000)
converter = WellLogToSpatial()
spatial_data = converter.convert(
    df,
    x=1000.0,
    y=2000.0,
    z='DEPTH',
    properties=['POROSITY', 'PERMEABILITY']
)

# Model permeability spatially
modeler = SpatialPropertyModeler(model_type='gpr')
modeler.fit_property(spatial_data, 'PERMEABILITY')

# Create prediction grid
grid = SpatialDataConverter.create_prediction_grid(
    x_range=(500, 1500),
    y_range=(1500, 2500),
    z_range=(0, 3000),
    nx=50, ny=50, nz=20
)

# Predict
predictions, uncertainty = modeler.predict(grid, return_std=True)
```

### Multi-Well Reservoir Model

```python
from geosuite.modeling import create_reservoir_model, SpatialDataConverter

# Load multiple wells
wells = {
    'Well_001': load_las_file('well_001.las').df(),
    'Well_002': load_las_file('well_002.las').df(),
    'Well_003': load_las_file('well_003.las').df(),
}

# Define well coordinates
coordinates = {
    'Well_001': (1000.0, 2000.0),
    'Well_002': (1500.0, 2500.0),
    'Well_003': (2000.0, 3000.0),
}

# Create reservoir model
model = create_reservoir_model(
    well_data=wells,
    coordinates=coordinates,
    properties=['POROSITY', 'PERMEABILITY'],
    model_type='gpr'
)

# Predict on grid
grid = SpatialDataConverter.create_prediction_grid(
    x_range=(0, 3000),
    y_range=(0, 4000),
    z_range=(0, 3000),
    nx=50, ny=50, nz=20
)

predictions = model.predict_all_properties(grid)
```

## Key Components

### WellLogToSpatial

Converts depth-based well log DataFrames to 3D spatial coordinates.

**Key Methods:**
- `convert()`: Convert single well to spatial format
- `convert_multiple_wells()`: Combine multiple wells into one spatial dataset

### SpatialPropertyModeler

Models reservoir properties spatially using Gaussian Process Regression.

**Key Methods:**
- `fit_permeability()`: Fit model for permeability
- `fit_porosity()`: Fit model for porosity
- `fit_property()`: Fit model for any property
- `predict()`: Predict property values at coordinates

### ReservoirModelBuilder

Builds complete 3D reservoir models with multiple properties.

**Key Methods:**
- `add_property_model()`: Add a property model
- `predict_all_properties()`: Predict all properties at once

### Workflow Functions

- `create_reservoir_model()`: High-level function to create multi-well models
- `interpolate_properties()`: Interpolate properties onto a grid

## Integration with GeoSuite Workflows

This module seamlessly integrates with other GeoSuite modules:

1. **I/O**: Load well logs using `geosuite.io`
2. **Petrophysics**: Calculate properties using `geosuite.petro`
3. **Modeling**: Convert to spatial and model using `geosuite.modeling`
4. **Visualization**: Plot results using `geosuite.plotting`

## Example Workflow

```python
# 1. Load and process well logs
from geosuite.io import load_las_file
from geosuite.petro import calculate_porosity_from_density, calculate_water_saturation

las = load_las_file('well.las')
df = las.df()

# 2. Calculate petrophysical properties
df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
df['WATER_SAT'] = calculate_water_saturation(df['POROSITY'], df['RT'])
df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4) / (df['WATER_SAT'] ** 2)

# 3. Convert to spatial format
from geosuite.modeling import WellLogToSpatial
converter = WellLogToSpatial()
spatial_data = converter.convert(df, x=1000, y=2000, z='DEPTH')

# 4. Model spatially
from geosuite.modeling import SpatialPropertyModeler
modeler = SpatialPropertyModeler()
modeler.fit_property(spatial_data, 'PERMEABILITY')

# 5. Predict on grid
from geosuite.modeling import SpatialDataConverter
grid = SpatialDataConverter.create_prediction_grid(
    (0, 3000), (0, 4000), (0, 5000)
)
predictions = modeler.predict(grid)
```

## References

- [pygeomodeling GitHub](https://github.com/kylejones200/pygeomodeling)
- [GeoSuite Documentation](https://geosuite.readthedocs.io)

