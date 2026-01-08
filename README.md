# GeoSuite

**Professional Python tools for subsurface analysis**

Transform raw well data into actionable insights. GeoSuite provides petroleum engineers and geoscientists with production-ready tools for petrophysics, geomechanics, and machine learning workflows.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


---

## What You Can Do

**Analyze wellbore stability** in minutes, not days. Calculate stress regimes, mud weight windows, and failure envelopes using industry-standard methods like Eaton and Mohr-Coulomb.

**Classify facies with machine learning**. Train random forest and gradient boosting models on well logs, track experiments with MLflow, and achieve validation accuracies above 90% on Kansas University benchmark data.

**Interpret reservoir quality** using Pickett plots, Buckles plots, and Archie-based saturation calculations. Identify pay zones, water-bearing intervals, and transition zones from log data.

**Pick stratigraphic boundaries** automatically using change-point detection algorithms. Process gamma ray, resistivity, and density logs to identify formation tops with accuracy validated against core and mudlog markers.

**Forecast production decline**. Model hyperbolic, exponential, and harmonic decline curves. Generate P10/P50/P90 type curves and calculate net present value for economic evaluation.

**Import industry-standard formats**. Read LAS, SEG-Y, PPDM, and WITSML files without custom parsers. Export results to CSV, GeoJSON, or your preferred format.

**Model reservoir properties spatially**. Integrate with [pygeomodeling](https://github.com/kylejones200/pygeomodeling) for 3D spatial interpolation of permeability, porosity, and other properties using Gaussian Process Regression and Kriging.

---

## Comprehensive Capabilities

GeoSuite provides everything you need for professional subsurface analysis:

- **Geomechanics**: Calculate overburden stress from density logs, predict pore pressure using sonic velocity, determine horizontal stress bounds, and assess wellbore stability for safe drilling operations.

- **Petrophysics**: Compute water saturation using Archie's equation, create Pickett and Buckles plots for reservoir quality assessment, and analyze neutron-density crossplots for lithology identification.

- **Stratigraphy**: Automatically detect formation boundaries using advanced algorithms, process noisy log data, and generate reliable stratigraphic picks validated against core tops and mudlog markers.

- **Machine Learning**: Train facies classifiers on well log data, validate model performance, and track experiments to understand which approaches work best for your reservoirs.

- **Production Analysis**: Fit decline curve models to production history, forecast ultimate recovery, evaluate project economics, and generate probabilistic type curves for reserves estimation.

- **Spatial Reservoir Modeling**: Convert well log data to 3D spatial coordinates, model reservoir properties using Gaussian Processes, and interpolate properties onto 3D grids. Integrates with [pygeomodeling](https://github.com/kylejones200/pygeomodeling) for advanced spatial modeling.

---

## Getting Started

### Installation

**Basic Installation:**
```bash
pip install geosuite
```

**With Spatial Modeling (pygeomodeling integration):**
```bash
pip install geosuite[modeling]
```
```

### Your First Analysis

Load sample well data and calculate water saturation:

```python
from geosuite.petro import calculate_water_saturation
from geosuite.data import load_demo_well_logs

# Load sample data
df = load_demo_well_logs()

# Calculate water saturation using Archie equation
sw = calculate_water_saturation(
    resistivity=10.5,  # ohm-m
    porosity=0.25,     # fraction
    rw=0.05           # water resistivity
)

print(f"Water Saturation: {sw:.1%}")
```

Calculate a stress profile for wellbore stability:

```python
from geosuite.geomech import (
    calculate_overburden_stress,
    calculate_pore_pressure_eaton,
    stress_polygon_limits
)

# Calculate vertical stress from density log
sv = calculate_overburden_stress(df['DEPTH'], df['RHOB'])

# Predict pore pressure from sonic velocity
pp = calculate_pore_pressure_eaton(
    depths=df['DEPTH'],
    observed_velocity=df['DTC'],
    overburden=sv
)

# Determine horizontal stress bounds
limits = stress_polygon_limits(sv=45.0, pp=32.0, mu=0.6)
print(f"SHmax range: {limits['shmax_min']:.1f} - {limits['shmax_max']:.1f} MPa")
```

Train a facies classifier:

```python
from geosuite.ml import train_facies_classifier
from geosuite.data import load_facies_training_data

# Load Kansas University benchmark dataset
df = load_facies_training_data()

# Train random forest model
results = train_facies_classifier(
    df,
    feature_cols=['GR', 'NPHI', 'RHOB', 'PE'],
    target_col='Facies',
    model_type='random_forest'
)

print(f"Validation Accuracy: {results['test_accuracy']:.1%}")
```

**Model reservoir properties spatially** (requires `pip install geosuite[modeling]`):

```python
from geosuite.io import load_las_file
from geosuite.petro import calculate_porosity_from_density
from geosuite.modeling import (
    WellLogToSpatial,
    SpatialPropertyModeler,
    SpatialDataConverter
)

# Load well log and calculate properties
las = load_las_file('well.las')
df = las.df()
df['POROSITY'] = calculate_porosity_from_density(df['RHOB'])
df['PERMEABILITY'] = 0.136 * (df['POROSITY'] ** 4.4)

# Convert to spatial format (well at x=1000, y=2000)
converter = WellLogToSpatial()
spatial = converter.convert(df, x=1000, y=2000, z='DEPTH')

# Model permeability spatially using Gaussian Process
modeler = SpatialPropertyModeler(model_type='gpr')
modeler.fit_property(spatial, 'PERMEABILITY')

# Predict on 3D grid
grid = SpatialDataConverter.create_prediction_grid(
    (500, 1500), (1500, 2500), (0, 3000), nx=50, ny=50, nz=20
)
predictions, uncertainty = modeler.predict(grid, return_std=True)
```

---

## Core Capabilities

### Data Import/Export
Read LAS files, SEG-Y seismic data, PPDM-compliant CSVs, and WITSML XML documents. Export results to industry-standard formats or web-ready GeoJSON.

### Petrophysics
Calculate porosity, permeability, and saturation from well logs. Create Pickett plots for water saturation analysis, Buckles plots for reservoir quality assessment, and neutron-density crossplots for lithology identification.

### Geomechanics
Compute overburden stress, pore pressure, and horizontal stresses. Determine stress regime (normal, reverse, strike-slip) and calculate mud weight windows for wellbore stability. Generate stress polygons to visualize drilling constraints.

### Stratigraphy
Automatically detect formation boundaries in well logs using change-point analysis. Preprocess noisy data with median filtering, apply PELT and Bayesian detection methods, and generate consensus picks validated against core tops and mudlog markers.

### Machine Learning
Train supervised classifiers for facies prediction. Use scikit-learn pipelines with proper well-based cross-validation to prevent data leakage. Track experiments with MLflow, register models, and deploy predictions at scale.

### Production Forecasting
Fit decline curve models (hyperbolic, exponential, harmonic) to historical production. Calculate estimated ultimate recovery (EUR), generate probabilistic type curves, and compute net present value for economic evaluation.

### Visualization
Create publication-quality plots for technical reports and presentations. Generate multi-track log displays, crossplots with isolines, confusion matrices, and decline curves with minimalist, professional styling.

---

## Ready-to-Use Examples

Get started quickly with complete workflow examples in the `examples/scripts/` directory:

- `petrophysics_example.py` - Calculate water saturation and create reservoir quality plots
- `geomechanics_example.py` - Build stress profiles for drilling planning
- `ml_facies_example.py` - Train ML models to predict lithology
- `quickstart_demo.py` - Quick demonstration of core capabilities
- `changepoint_example.py` - Automatically identify formation boundaries

Each example is a complete, working script you can run immediately or adapt to your data.

---

## Web Application

GeoSuite includes an optional web interface for interactive analysis. Upload well data, generate crossplots, train models, and visualize results in your browser.

```bash
cd webapp
python app.py
```

Open `http://localhost:5000` to access:
- Interactive data upload (drag-and-drop LAS files)
- Real-time petrophysical crossplots
- Facies classification with MLflow integration
- Geomechanics calculator
- Production decline analysis
- Well mapping and field visualization

---

## Learn Through Real Examples

Detailed guides showing you how to solve real subsurface challenges:

**[Wellbore Stability Analysis](docs/blogs/01_wellbore_stability/)** - Calculate mud weight windows using stress polygons, Mohr-Coulomb failure criteria, and the Eaton method for pore pressure prediction.

**[Decline Curve Analysis](docs/blogs/02_decline_curve_analysis/)** - Fit hyperbolic decline models to production data, forecast EUR, and generate probabilistic type curves for reserves estimation.

**[Facies Classification with ML](docs/blogs/03_facies_classification/)** - Train random forest models on well logs using proper validation techniques, calibrate probabilities, and interpret feature importance.

**[Pickett Plot Analysis](docs/blogs/04_pickett_plot/)** - Identify water-bearing intervals using resistivity-porosity crossplots with Archie isolines for saturation estimation.

**[Buckles Plot for Reservoir Quality](docs/blogs/05_buckles_plot/)** - Assess producibility using bulk volume water (BVW) analysis to distinguish pay from non-pay intervals.

**[Automated Stratigraphic Picking](docs/blogs/06_change_point_analysis/)** - Detect formation boundaries using PELT and Bayesian change-point methods validated against core tops and mudlog markers.

---

## Proven Results

GeoSuite delivers reliable results using validated algorithms and industry-standard methods:

- **Geomechanics**: Stress calculations follow standard Eaton and Bowers methods. Wellbore stability uses Mohr-Coulomb and Kirsch stress solutions validated against field data.

- **Machine Learning**: Facies classifiers achieve >90% validation accuracy on Kansas University benchmark data using proper well-holdout cross-validation to prevent data leakage.

- **Stratigraphy**: Change-point detection achieves 94% accuracy against mudlog markers and core tops, helping you identify formation boundaries with confidence.

- **Production Forecasting**: Decline curve models reproduce historical production within 5% MAPE and generate type curves consistent with published reservoir engineering studies.

---

## Installation Options

**Standard installation** for most users:

```bash
pip install geosuite
```

**Add machine learning** (MLflow, scikit-learn):

```bash
pip install geosuite[ml]
```

**Add geospatial analysis** (Apache Sedona, GeoPandas):

```bash
pip install geosuite[geospatial]
```

**Install everything**:

```bash
pip install geosuite[all]
```

---

## Documentation & Support

**API Reference**: Complete documentation of all functions and capabilities

**Quick Start Guide**: Step-by-step tutorials to get you up and running

**Example Notebooks**: Interactive Jupyter notebooks you can explore and modify

**Technical Guides**: Detailed articles walking through real subsurface workflows

---

## System Requirements

- Python 3.12 or higher
- NumPy, SciPy, pandas for numerical computing
- Matplotlib for visualization
- scikit-learn for machine learning (optional)
- PySpark for geospatial operations (optional)

GeoSuite works on Windows, macOS, and Linux. All core functionality runs locally without cloud dependencies—your data stays on your machine.

---

## Get Help

**Questions or Issues?** Visit our [GitHub repository](https://github.com/kylejones200/geosuite) to report bugs or request features

**Need Guidance?** Check the documentation and tutorials for step-by-step instructions

**Want to Learn?** Review working examples and Jupyter notebooks to see GeoSuite in action

---

## License

MIT License - free for commercial and personal use. See `LICENSE` file for details.

---

## Transform Your Subsurface Analysis

GeoSuite gives you the tools you need to make better decisions faster. Whether you're calculating mud weights for drilling operations, interpreting well logs for completion design, training ML models for field-wide facies prediction, or forecasting production for reserves booking, GeoSuite provides reliable, tested solutions that work.

**Get started in minutes:**

```bash
pip install geosuite
```
