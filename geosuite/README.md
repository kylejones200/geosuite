# GeoSuite

A unified, comprehensive application for geomechanics, subsurface geology, and machine learning workflows in petroleum engineering and geoscience.

## Overview

GeoSuite is a Flask-based web application that brings together multiple geoscience projects into one integrated platform. It provides tools for data import, petrophysical analysis, facies classification with machine learning, geomechanics calculations, and production data analysis.

## Features

### Data Management
- **Multi-format Import**: LAS files, CSV, SEG-Y (with segyio)
- **PPDM Support**: Industry-standard PPDM 3NF data structures
- **WITSML Parsing**: Well data standards integration
- **Demo Datasets**: Built-in datasets for testing and learning

### Petrophysics
- Pickett Plot analysis
- Buckles Plot for reservoir quality
- Multi-well log crossplots
- Interactive visualization with Plotly

### Machine Learning & Facies Classification
- **Multiple Algorithms**: 
  - Support Vector Machines (SVM)
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
- **MLflow Integration**: Full experiment tracking and model registry
- **Confusion Matrix Utilities**: Advanced metrics with adjacent facies support
- **Pre-loaded Datasets**: 
  - Kansas University facies dataset (9 training wells)
  - Validation datasets for model testing
  - Real well log data with expert labels

### Geomechanics
- Overburden stress (Sv) calculations
- Hydrostatic pressure profiles
- Pore pressure estimation
- Effective stress analysis
- SHmax bounds from wellbore failure

### Wells & Production Analysis
- Interactive well mapping
- Decline Curve Analysis (DCA)
- Production forecasting
- Operator performance benchmarking
- Cut analysis (oil/water, GOR)
- Bakken formation analysis with 23K+ wells

## Quick Start

### Installation

1. **Clone or navigate to the repository**
```bash
cd /Users/k.jones/Documents/geos/geosuite
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

## Project Structure

```
geosuite/
├── app/                          # Flask application
│   ├── blueprints/              # Modular route blueprints
│   │   ├── api/                 # API endpoints
│   │   ├── data/                # Data import & management
│   │   ├── geomech/             # Geomechanics calculations
│   │   ├── main/                # Main pages
│   │   ├── ml/                  # Machine learning & MLflow
│   │   ├── petro/               # Petrophysics tools
│   │   └── wells/               # Wells & production analysis
│   ├── services/                # Business logic layer
│   │   ├── databricks_service.py
│   │   └── mlflow_service.py
│   ├── static/                  # CSS, JavaScript, images
│   └── templates/               # HTML templates
├── geosuite_lib/                # Core Python libraries
│   ├── data/                    # Data loaders & demo datasets
│   │   ├── files/              # CSV datasets (facies, wells, etc.)
│   │   └── demo_datasets.py    # Dataset loading functions
│   ├── geomech/                # Geomechanics calculators
│   ├── io/                     # Data import/export
│   │   ├── las_loader.py
│   │   ├── segy_loader.py
│   │   ├── ppdm_parser.py
│   │   └── witsml_parser.py
│   ├── ml/                     # Machine learning
│   │   ├── classifiers.py
│   │   ├── enhanced_classifiers.py
│   │   └── confusion_matrix_utils.py
│   └── petro/                  # Petrophysics
│       └── archie.py
├── notebooks_archive/           # Jupyter notebooks
│   ├── Facies Classification - SVM.ipynb
│   ├── Building a Geomechanical Model.ipynb
│   ├── Las loader.ipynb
│   ├── Read Segy Data.ipynb
│   ├── Change-Point Analysis.ipynb
│   └── ... (more notebooks)
├── mlruns/                      # MLflow experiment tracking
├── mlartifacts/                 # MLflow model artifacts
├── sample_ppdm/                 # Sample PPDM data
├── sample_witsml/               # Sample WITSML data
├── app.py                       # Application entry point
├── mlflow_config.py            # MLflow configuration
└── requirements.txt            # Python dependencies
```

## Integrated Projects

GeoSuite brings together functionality from multiple geoscience projects:

### 1. **Facies Classification (Brendon Hall)**
- SVM-based facies prediction from well logs
- University of Kansas training datasets
- Confusion matrix utilities with adjacent facies support
- Source: `facies_classification-master/`

### 2. **GeoHacker Notebooks**
- LAS file loading and analysis
- SEG-Y seismic data reading
- Change-point analysis
- Well log data visualization
- Source: `GeoHacker-master/notebooks/`

### 3. **Production Analytics**
- Bakken well analysis
- PPDM data structures
- Decline curve analysis
- Operator performance metrics

### 4. **MLflow Experiment Tracking**
- Model versioning and registry
- Experiment comparison
- Hyperparameter tracking
- Deployment-ready model serving

## Demo Mode (No Uploads Required)

GeoSuite includes built-in demo datasets so you can explore all features without uploading data:

### Data Import (`/data`)
- Click "Load demo well logs" to preview sample LAS-style data
- Click "Load demo facies" to see facies-labeled logs

### Petrophysics (`/petro`)
- Click "Load demo well logs" for RHOB vs NPHI crossplot
- Interactive selection and filtering

### Facies Classification (`/ml`)
- Click "Load demo facies dataset" for training data
- Train models with different algorithms
- Compare model performance

### Geomechanics (`/geomech`)
- Click "Load demo geomech profiles" 
- View Sv, hydrostatic pressure, and effective stress

### Wells Analysis (`/wells`)
- Automatic loading of 23K+ Bakken wells
- Click any well on map for detailed analysis
- View decline curves and production history

## Available Datasets

All datasets are located in `geosuite_lib/data/files/`:

### Facies Classification
- `training_data.csv` - 9 wells with expert facies labels
- `validation_data_nofacies.csv` - Unlabeled test data
- `facies_vectors.csv` - Complete facies vectors
- `well_data_with_facies.csv` - Full well log suite
- `KSdata/training_wells.csv` - Kansas training set
- `KSdata/test_wells.csv` - Kansas test set

### Well & Production Data
- `demo_well_logs.csv` - Sample well log data
- `demo_facies.csv` - Small facies demo
- `field_data.csv` - Multi-well field data

### Loading Datasets in Python

```python
from geosuite_lib.data import demo_datasets

# Load facies training data
df_train = demo_datasets.load_facies_training_data()

# Load Kansas wells
df_ks_train = demo_datasets.load_kansas_training_wells()
df_ks_test = demo_datasets.load_kansas_test_wells()

# Load demo logs
df_logs = demo_datasets.load_demo_well_logs()
```

## Machine Learning Workflow

### Training a Facies Classifier

```python
from geosuite_lib.ml.enhanced_classifiers import MLflowFaciesClassifier
from geosuite_lib.data import demo_datasets

# Load data
df = demo_datasets.load_facies_training_data()

# Initialize classifier with MLflow tracking
classifier = MLflowFaciesClassifier()

# Define features and target
feature_cols = ['GR', 'NPHI', 'RHOB', 'PE', 'DEPTH']
target_col = 'Facies'

X = df[feature_cols]
y = df[target_col]

# Train model
results = classifier.train_model(
    X=X, 
    y=y, 
    model_type='random_forest',
    n_estimators=100,
    max_depth=10
)

print(f"Test Accuracy: {results['test_accuracy']:.3f}")

# Register model
classifier.register_model(
    model_name='geosuite-facies-rf',
    description='Random Forest facies classifier'
)
```

### Viewing Confusion Matrix

```python
from geosuite_lib.ml.confusion_matrix_utils import display_cm, compute_metrics_from_cm

# Display confusion matrix with metrics
display_cm(
    cm=results['confusion_matrix'],
    labels=results['classes'],
    display_metrics=True
)

# Get metrics as DataFrame
metrics_df = compute_metrics_from_cm(
    cm=results['confusion_matrix'],
    labels=results['classes']
)
print(metrics_df)
```

## MLflow Integration

GeoSuite uses MLflow for comprehensive ML experiment tracking:

### Features
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and stage models (Staging, Production)
- **Model Comparison**: Compare multiple model runs
- **Reproducibility**: Full experiment history and lineage

### Accessing MLflow UI

```bash
mlflow ui --port 5001
```

Then open: `http://localhost:5001`

### Web Interface
Navigate to `/ml` in GeoSuite to:
- View all experiments
- Compare model performance
- Register and promote models
- Deploy models for prediction

## API Endpoints

### ML API
- `POST /ml/api/train-facies-model` - Train new model
- `GET /ml/api/experiments` - List experiments
- `GET /ml/api/models` - List registered models
- `POST /ml/api/register-model` - Register model
- `POST /ml/api/promote-model` - Promote model stage
- `POST /ml/api/load-model` - Load model for prediction

### Data API
- `POST /api/upload-las` - Upload LAS file
- `POST /api/upload-csv` - Upload CSV data
- `GET /api/demo-data` - Get demo datasets

### Wells API
- `GET /wells/api/wells-geojson` - Get wells as GeoJSON
- `GET /wells/api/well-production/{well_id}` - Get production data
- `GET /wells/api/operators` - List operators

## Development

### Adding New Features

1. Create a new blueprint in `app/blueprints/`
2. Add core logic to `geosuite_lib/`
3. Create templates in `app/templates/`
4. Register blueprint in `app/__init__.py`

### Testing

```bash
# Run tests (when available)
pytest

# Test MLflow integration
python geosuite_lib/ml/enhanced_classifiers.py
```

## Documentation

- `BAKKEN_ANALYSIS_README.md` - Bakken wells analysis guide
- `CLEANUP_SUMMARY.md` - Project consolidation history

## Roadmap

### Completed
- Flask multipage application structure
- Data import (LAS, CSV)
- MLflow integration
- Facies classification with multiple algorithms
- Confusion matrix utilities
- Kansas University facies datasets
- GeoHacker notebooks integration
- Bakken wells analysis
- PPDM data structures
- Production analytics

### In Progress
- Enhanced SEG-Y visualization
- Advanced petrophysical plots
- Geomechanics calculators from notebooks
- Unit tests and integration tests

### Planned
- Docker deployment
- Database backend (PostgreSQL)
- User authentication
- API documentation (Swagger/OpenAPI)
- Databricks integration
- Real-time well monitoring
- Cloud deployment (AWS/Azure)

## Contributing

GeoSuite consolidates multiple open-source geoscience projects. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project integrates code from multiple sources:
- Facies Classification by Brendon Hall
- GeoHacker project notebooks
- Original GeoSuite development

Please refer to individual project licenses where applicable.

## Acknowledgments

- **Brendon Hall** - Facies classification SVM implementation
- **University of Kansas** - Facies training datasets
- **Software Underground** - GeoHacker initiative and community
- **MLflow Community** - ML experiment tracking framework

## Contact

For questions, issues, or contributions, please open an issue on the repository.

---
