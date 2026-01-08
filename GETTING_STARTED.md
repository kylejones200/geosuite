# Getting Started with GeoSuite

GeoSuite is a comprehensive Python library for geoscience workflows, providing tools for petrophysics, geomechanics, machine learning, and stratigraphic analysis.

## Installation

### Basic Installation

```bash
pip install geosuite
```

### With Optional Dependencies

For specific features, install with extras:

```bash
# For WITSML support
pip install geosuite[witsml]

# For spatial modeling (pygeomodeling)
pip install geosuite[modeling]

# For ML interpretability (SHAP)
pip install geosuite[ml]

# For all optional features
pip install geosuite[all]
```

## Quick Start

### 1. Load Well Log Data

```python
import geosuite
from geosuite.data import load_demo_well_logs

# Load demo well log data
df = load_demo_well_logs()
print(df.head())
```

### 2. Petrophysics Calculations

#### Water Saturation (Archie)

```python
from geosuite.petro import calculate_water_saturation

# Calculate water saturation
sw = calculate_water_saturation(
    phi=df['PHIE'],  # Porosity
    rt=df['RESDEEP'],  # Deep resistivity
    rw=0.05,  # Water resistivity
    m=2.0,  # Cementation exponent
    n=2.0  # Saturation exponent
)

df['SW'] = sw
```

#### Porosity from Density

```python
from geosuite.petro import calculate_porosity_from_density

phi = calculate_porosity_from_density(
    rhob=df['RHOB'],  # Bulk density
    rho_matrix=2.65,  # Matrix density (g/cc)
    rho_fluid=1.0  # Fluid density (g/cc)
)
```

#### Permeability Estimation

```python
from geosuite.petro.permeability import calculate_permeability_kozeny_carman

perm = calculate_permeability_kozeny_carman(
    phi=df['PHIE'],
    sw=df['SW'],
    grain_size_microns=100.0
)
```

### 3. Geomechanics Calculations

#### Overburden Stress

```python
from geosuite.geomech import calculate_overburden_stress

sv = calculate_overburden_stress(
    depth=df['DEPTH'],
    rhob=df['RHOB'],
    g=9.81
)
```

#### Pore Pressure (Eaton Method)

```python
from geosuite.geomech import calculate_pore_pressure_eaton, calculate_hydrostatic_pressure

# Calculate hydrostatic pressure first
ph = calculate_hydrostatic_pressure(df['DEPTH'])

# Calculate pore pressure using Eaton method
pp = calculate_pore_pressure_eaton(
    depth=df['DEPTH'],
    dt=df['DTC'],  # Compressional transit time
    dt_normal=df['DTC'].quantile(0.5),  # Normal trend
    sv=sv,
    ph=ph,
    exponent=3.0
)
```

#### Effective Stress

```python
from geosuite.geomech import calculate_effective_stress

sigma_eff = calculate_effective_stress(
    sv=sv,
    pp=pp,
    biot=1.0
)
```

### 4. Stratigraphic Analysis

#### Change-Point Detection

```python
from geosuite.stratigraphy import preprocess_log, detect_pelt, detect_bayesian_online

# Preprocess log data
gr_clean = preprocess_log(
    df['GR'].values,
    median_window=5,
    detrend_window=100
)

# Detect change points using PELT algorithm
changepoints_pelt = detect_pelt(
    gr_clean,
    penalty=10.0,
    model='l2'
)

# Or use Bayesian online detection
changepoints_bayesian = detect_bayesian_online(
    gr_clean,
    hazard=100.0
)
```

### 5. Machine Learning

#### Facies Classification

```python
from geosuite.ml import train_and_predict

# Train a facies classifier
results = train_and_predict(
    df=df,
    feature_cols=['GR', 'NPHI', 'RHOB', 'PE'],
    target_col='Facies',
    model_type='random_forest',
    test_size=0.2
)

print(results.report)
```

#### Permeability Prediction

```python
from geosuite.ml import PermeabilityPredictor

# Create predictor
predictor = PermeabilityPredictor(model_type='random_forest')

# Fit on training data
predictor.fit(
    X_train[['PHIE', 'SW', 'GR', 'RHOB']],
    y_train  # Known permeability
)

# Predict on new data
permeability_pred = predictor.predict(X_test[['PHIE', 'SW', 'GR', 'RHOB']])
```

#### Clustering for Facies Grouping

```python
from geosuite.ml import FaciesClusterer

# Create clusterer
clusterer = FaciesClusterer(
    method='kmeans',
    n_clusters=5,
    scale_features=True
)

# Fit and predict
cluster_labels = clusterer.fit_predict(
    df[['GR', 'NPHI', 'RHOB', 'PE']]
)

df['Facies_Cluster'] = cluster_labels
```

### 6. Visualization

#### Pickett Plot

```python
from geosuite.petro import pickett_plot
import matplotlib.pyplot as plt

fig = pickett_plot(
    df,
    porosity_col='PHIE',
    resistivity_col='RESDEEP',
    m=2.0,
    n=2.0
)
plt.show()
```

#### Strip Chart

```python
from geosuite.plotting import create_strip_chart

fig = create_strip_chart(
    df,
    depth_col='DEPTH',
    log_cols=['GR', 'NPHI', 'RHOB', 'RESDEEP'],
    facies_col='Facies'
)
plt.show()
```

### 7. Complete Workflow Example

```python
import geosuite
from geosuite.data import load_demo_well_logs
from geosuite.petro import calculate_water_saturation, calculate_porosity_from_density
from geosuite.geomech import (
    calculate_overburden_stress,
    calculate_hydrostatic_pressure,
    calculate_pore_pressure_eaton,
    create_pressure_dataframe
)
from geosuite.ml import train_and_predict

# Load data
df = load_demo_well_logs()

# Petrophysics
df['PHIE'] = calculate_porosity_from_density(df['RHOB'])
df['SW'] = calculate_water_saturation(
    phi=df['PHIE'],
    rt=df['RESDEEP'],
    rw=0.05
)

# Geomechanics
df['SV'] = calculate_overburden_stress(df['DEPTH'], df['RHOB'])
df['PH'] = calculate_hydrostatic_pressure(df['DEPTH'])

# Create comprehensive pressure DataFrame
pressure_df = create_pressure_dataframe(
    depth=df['DEPTH'],
    rhob=df['RHOB'],
    sv=df['SV'],
    ph=df['PH']
)

# Machine Learning - Facies Classification
results = train_and_predict(
    df=df,
    feature_cols=['GR', 'NPHI', 'RHOB', 'PE'],
    target_col='Facies',
    model_type='random_forest'
)

print("Workflow complete!")
print(f"Water saturation range: {df['SW'].min():.2f} - {df['SW'].max():.2f}")
print(f"Overburden stress range: {df['SV'].min():.1f} - {df['SV'].max():.1f} MPa")
```

## Advanced Features

### Uncertainty Quantification

```python
from geosuite.utils.uncertainty import uncertainty_porosity_from_density

# Calculate porosity with uncertainty
phi_result = uncertainty_porosity_from_density(
    rhob=df['RHOB'],
    rho_matrix=2.65,
    rho_fluid=1.0,
    rhob_std=0.05,  # Uncertainty in density
    method='monte_carlo',
    n_samples=1000
)

print(f"Porosity: {phi_result['mean']:.3f} ± {phi_result['std']:.3f}")
```

### Spatial Reservoir Modeling

```python
from geosuite.modeling import SpatialPropertyModeler

# Create modeler
modeler = SpatialPropertyModeler(model_type='gpr')

# Fit permeability model
modeler.fit_permeability(
    spatial_data=spatial_df,
    property_col='PERMEABILITY'
)

# Predict at new locations
predictions = modeler.predict(
    coords=new_coords
)
```

### Configuration Management

```python
from geosuite.config import load_config, get_config

# Load configuration from YAML
load_config('config.yaml')

# Get configuration values
rw = get_config('petrophysics.archie.rw', default=0.05)
m = get_config('petrophysics.archie.m', default=2.0)
```

## Next Steps

- Explore the [API Documentation](docs/source/api/index.rst)
- Check out [Example Notebooks](examples/)
- Review [Advanced Guides](docs/source/guides/)
- See the [Roadmap](roadmap.md) for upcoming features

## Getting Help

- **Documentation**: https://github.com/kylejones200/geosuite
- **Issues**: Report bugs or request features on GitHub
- **Examples**: See the `examples/` directory for more workflows

