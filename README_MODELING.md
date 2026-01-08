# GeoSuite + pygeomodeling Integration

GeoSuite now integrates with [pygeomodeling](https://github.com/kylejones200/pygeomodeling) for advanced spatial reservoir property modeling using Gaussian Process Regression and Kriging.

## What This Enables

- **Spatial Interpolation**: Model reservoir properties (permeability, porosity) in 3D space
- **Multi-Well Integration**: Combine data from multiple wells into unified reservoir models
- **Uncertainty Quantification**: Get uncertainty estimates with every prediction
- **3D Grid Prediction**: Interpolate properties onto regular 3D grids for visualization

## Installation

```bash
# Install with modeling support
pip install geosuite[modeling]

# Or install separately
pip install pygeomodeling
```

## Quick Example

```python
from geosuite.io import load_las_file
from geosuite.petro import calculate_porosity_from_density
from geosuite.modeling import (
    WellLogToSpatial,
    SpatialPropertyModeler,
    SpatialDataConverter
)

# 1. Load well log
las = load_las_file('well.las')
df = las.df()

# 2. Calculate properties
df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4)

# 3. Convert to spatial (well at x=1000, y=2000)
converter = WellLogToSpatial()
spatial = converter.convert(df, x=1000, y=2000, z='DEPTH')

# 4. Model spatially
modeler = SpatialPropertyModeler()
modeler.fit_property(spatial, 'PERMEABILITY')

# 5. Predict on 3D grid
grid = SpatialDataConverter.create_prediction_grid(
    (500, 1500), (1500, 2500), (0, 3000), nx=50, ny=50, nz=20
)
predictions, uncertainty = modeler.predict(grid, return_std=True)
```

## Multi-Well Reservoir Model

```python
from geosuite.modeling import create_reservoir_model

# Load multiple wells
wells = {
    'Well_001': load_las_file('well_001.las').df(),
    'Well_002': load_las_file('well_002.las').df(),
}

# Define coordinates
coords = {
    'Well_001': (1000, 2000),
    'Well_002': (1500, 2500),
}

# Create model
model = create_reservoir_model(
    wells, coords, properties=['POROSITY', 'PERMEABILITY']
)

# Predict all properties
grid = SpatialDataConverter.create_prediction_grid(...)
predictions = model.predict_all_properties(grid)
```

## Documentation

See `geosuite/modeling/README.md` for detailed documentation.

## References

- [pygeomodeling](https://github.com/kylejones200/pygeomodeling) - Advanced GP and Kriging toolkit
- [GeoSuite Documentation](https://geosuite.readthedocs.io)

