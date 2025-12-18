# GeoSuite Web Application

**Interactive subsurface analysis in your browser**

Access all GeoSuite capabilities through a user-friendly web interface. No command line required—just upload your data and start analyzing.

## What You Get

- **Interactive Data Upload**: Drag and drop LAS, CSV, or SEG-Y files
- **Real-time Visualization**: Create petrophysical crossplots and well logs instantly
- **Machine Learning**: Train and evaluate facies classifiers with a few clicks
- **Geomechanics Calculator**: Calculate stress and pressure profiles visually
- **Production Analysis**: Generate decline curves and map your wells
- **Model Tracking**: Keep track of your machine learning experiments

## Quick Start

### Install

First, install GeoSuite and the web app dependencies:

```bash
pip install geosuite
pip install -r requirements-webapp.txt
```

Or install everything at once:

```bash
pip install geosuite[webapp]
```

### Run

Start the web application:

```bash
cd webapp
python app.py
```

Then open your browser to: `http://localhost:5000`

That's it! You're ready to start analyzing your well data.

## What You Can Do

### Manage Your Data
- Upload LAS files, CSV data, or SEG-Y seismic
- Load demo datasets to try out features
- Preview and validate your data before analysis
- Export processed results in multiple formats

### Analyze Petrophysics
- Create Pickett plots to identify water saturation
- Generate Buckles plots for reservoir quality assessment
- Build neutron-density crossplots for lithology identification
- Display multi-track well logs interactively

### Calculate Geomechanics
- Calculate overburden stress from density logs
- Estimate pore pressure using sonic velocity
- Generate stress polygons to assess wellbore stability
- Visualize pressure profiles for drilling planning

### Apply Machine Learning
- Train facies classifiers on your well log data
- Compare multiple algorithms (Random Forest, SVM, Gradient Boosting, Logistic Regression)
- View confusion matrices to understand model performance
- Track experiments and register your best models

### Visualize Production
- Map your wells on an interactive map
- Generate decline curve forecasts
- Analyze operator performance across fields
- View field-level production analytics

### Access Programmatically
- Use RESTful API endpoints for automation
- Upload data and train models via API
- Integrate with your existing workflows

## Configuration

The web application works out of the box with sensible defaults. For custom setups, you can optionally configure:

- **File Upload Limits**: Adjust maximum file sizes for your datasets
- **Data Storage**: Configure where uploaded files are stored
- **External Databases**: Connect to PostgreSQL or other databases (optional)
- **Model Tracking**: Specify where machine learning experiments are saved

## Getting Started

### Analyze Your Well Logs

1. Open the web app at `http://localhost:5000`
2. Navigate to **Data** and upload your LAS file
3. Preview your well log data
4. Go to **Petrophysics** and create crossplots
5. Identify pay zones and reservoir quality

### Train a Machine Learning Model

1. Navigate to **Machine Learning**
2. Load the demo facies dataset (or upload your own)
3. Select which log curves to use as features
4. Choose your algorithm and click **Train Model**
5. View results and see how accurate your model is
6. Save your model for future use

### Calculate Drilling Stresses

1. Navigate to **Geomechanics**
2. Upload well log data with density and sonic curves
3. Enter your formation parameters
4. Calculate overburden stress and pore pressure
5. View the stress profile to plan mud weights
6. Export results for your drilling plan

## Automate with the API

You can automate your workflows using the built-in REST API:

```python
import requests

# Upload your data
files = {'file': open('well_log.las', 'rb')}
response = requests.post('http://localhost:5000/api/upload-las', files=files)

# Train a model programmatically
payload = {
    'features': ['GR', 'NPHI', 'RHOB'],
    'target': 'Facies',
    'model_type': 'random_forest'
}
response = requests.post('http://localhost:5000/ml/api/train-facies-model', json=payload)
```

## Track Your ML Experiments

The web app includes MLflow integration to help you keep track of model training experiments. Access the MLflow dashboard to:

- Compare different model configurations
- Review accuracy metrics over time
- Register and version your best models
- Document which models work best for your data

## Learn More

- **GeoSuite Python Library**: See the main README for the core library
- **Code Examples**: Check out the `examples/` directory for working scripts
- **Blog Posts**: Read detailed technical articles in `docs/blogs/`

## Common Questions

**The app won't start**  
Make sure you've installed all dependencies with `pip install -r requirements-webapp.txt`

**My file won't upload**  
Check that your file is under 16MB. For larger datasets, consider using the Python library directly.

**I'm getting calculation errors**  
Verify your data has the required log curves (e.g., RHOB for density, DT for sonic). The app will let you know which curves are needed.

## Support

Questions or issues? Check out the [GitHub repository](https://github.com/kylejones200/geosuite) for documentation and support.

---

