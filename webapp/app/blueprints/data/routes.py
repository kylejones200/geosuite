"""
Data routes for file import, management, and processing.
"""

from flask import render_template, request, jsonify, current_app
from . import bp
import sys
import os
import pandas as pd

# Add the geosuite_lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'geosuite_lib'))

try:
    from io.csv_loader import load_csv_data
    from io.las_loader import load_las_data
    from data.demo_datasets import get_demo_well_logs, get_demo_facies_data
except ImportError:
    # Fallback functions if modules not available
    def load_csv_data(filepath):
        return pd.read_csv(filepath)
    
    def load_las_data(filepath):
        return {"error": "LAS loader not available"}
    
    def get_demo_well_logs():
        return pd.DataFrame({
            'DEPTH': range(1000, 2000, 10),
            'GR': [50 + i * 0.1 for i in range(100)],
            'NPHI': [0.2 + i * 0.001 for i in range(100)],
            'RHOB': [2.5 + i * 0.002 for i in range(100)]
        })
    
    def get_demo_facies_data():
        return pd.DataFrame({
            'DEPTH': range(1000, 1500, 5),
            'FACIES': ['Sandstone', 'Shale'] * 50
        })


@bp.route("/")
def data_home():
    """Data management home page."""
    return render_template("data/index.html")


@bp.route("/import")
def data_import():
    """Data import page."""
    return render_template("data/import.html")


@bp.route("/api/demo-datasets")
def get_demo_datasets():
    """Get available demo datasets."""
    try:
        datasets = {
            "well_logs": {
                "name": "Demo Well Logs",
                "description": "Synthetic well log data (GR, NPHI, RHOB)",
                "type": "logs",
                "format": "CSV"
            },
            "facies": {
                "name": "Demo Facies Data", 
                "description": "Facies classification training data",
                "type": "classification",
                "format": "CSV"
            }
        }
        
        return jsonify({
            "status": "success",
            "datasets": datasets
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to get demo datasets",
            "error": str(e)
        }), 500


@bp.route("/api/load-demo/<dataset_type>")
def load_demo_dataset(dataset_type):
    """Load a specific demo dataset."""
    try:
        if dataset_type == "well_logs":
            df = get_demo_well_logs()
        elif dataset_type == "facies":
            df = get_demo_facies_data()
        else:
            return jsonify({
                "status": "error",
                "message": "Unknown dataset type",
                "available": ["well_logs", "facies"]
            }), 400
        
        return jsonify({
            "status": "success",
            "dataset_type": dataset_type,
            "data": df.head(100).to_dict(orient="records"),
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_stats": {
                "depth_range": [float(df['DEPTH'].min()), float(df['DEPTH'].max())] if 'DEPTH' in df.columns else None,
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist()
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to load demo dataset",
            "error": str(e)
        }), 500


@bp.route("/api/upload-file", methods=["POST"])
def upload_file():
    """Upload and process data files."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
            
            # Basic data validation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            return jsonify({
                "status": "success",
                "message": "CSV file processed successfully",
                "filename": file.filename,
                "rows": len(df),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "text_columns": text_cols,
                "sample_data": df.head(5).to_dict(orient="records"),
                "data_types": df.dtypes.astype(str).to_dict()
            })
            
        elif filename.endswith('.las'):
            # Placeholder for LAS file processing
            return jsonify({
                "status": "partial",
                "message": "LAS file detected - advanced processing available",
                "filename": file.filename,
                "note": "Full LAS processing requires lasio library"
            })
            
        else:
            return jsonify({
                "status": "error",
                "message": "Unsupported file type",
                "supported_formats": [".csv", ".las"],
                "filename": file.filename
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "File upload failed",
            "error": str(e)
        }), 500


@bp.route("/ppdm")
def ppdm_management():
    """PPDM data management page."""
    return render_template("data/ppdm.html")


@bp.route("/api/ppdm-validate", methods=["POST"])
def validate_ppdm():
    """Validate PPDM data format."""
    try:
        data = request.json
        file_type = data.get('file_type', 'unknown')
        
        # Basic PPDM validation (placeholder)
        required_fields = {
            'wells': ['UWI', 'WELL_NAME', 'LATITUDE', 'LONGITUDE'],
            'production': ['UWI', 'PRODUCTION_DATE', 'OIL_VOLUME', 'GAS_VOLUME'],
            'business_associates': ['BA_ID', 'BA_NAME', 'BA_TYPE']
        }
        
        if file_type in required_fields:
            return jsonify({
                "status": "success",
                "file_type": file_type,
                "required_fields": required_fields[file_type],
                "validation": "PPDM format requirements identified"
            })
        else:
            return jsonify({
                "status": "warning",
                "message": "Unknown PPDM file type",
                "supported_types": list(required_fields.keys())
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "PPDM validation failed",
            "error": str(e)
        }), 400


@bp.route("/strip-chart")
def strip_chart():
    """Strip chart visualization page."""
    return render_template("data/strip_chart.html")


@bp.route("/api/strip-chart-data")
def get_strip_chart_data():
    """
    Generate strip chart data for visualization.
    
    Query parameters:
        - source: 'demo' or 'uploaded' (default: 'demo')
        - logs: comma-separated list of log names (e.g., 'GR,RHOB,NPHI')
        - depth_min: minimum depth (optional)
        - depth_max: maximum depth (optional)
    """
    try:
        source = request.args.get('source', 'demo')
        log_names = request.args.get('logs', 'GR,RHOB,NPHI').split(',')
        depth_min = request.args.get('depth_min', type=float)
        depth_max = request.args.get('depth_max', type=float)
        
        # Load data based on source
        if source == 'demo':
            from geosuite_lib.data import demo_datasets
            df = demo_datasets.load_demo_well_logs()
        else:
            # TODO: Implement uploaded data loading
            return jsonify({
                "status": "error",
                "message": "Uploaded data not yet implemented"
            }), 400
        
        # Create strip chart
        from geosuite_lib.plotting.strip_charts import create_strip_chart
        
        depth_range = None
        if depth_min is not None and depth_max is not None:
            depth_range = (depth_min, depth_max)
        
        fig = create_strip_chart(
            df=df,
            log_cols=log_names,
            depth_range=depth_range,
            title='Well Log Strip Chart'
        )
        
        return jsonify({
            "status": "success",
            "figure": fig.to_json(),
            "n_samples": len(df),
            "logs": log_names
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to generate strip chart",
            "error": str(e)
        }), 500


@bp.route("/api/facies-strip-chart")
def get_facies_strip_chart():
    """
    Generate strip chart with facies classification overlay.
    
    Query parameters:
        - dataset: which facies dataset to load (default: 'training')
        - well: specific well name (optional)
    """
    try:
        dataset_type = request.args.get('dataset', 'training')
        well_name = request.args.get('well')
        
        # Load facies data
        from geosuite_lib.data import demo_datasets
        
        if dataset_type == 'training':
            df = demo_datasets.load_facies_training_data()
        elif dataset_type == 'kansas_train':
            df = demo_datasets.load_kansas_training_wells()
        elif dataset_type == 'kansas_test':
            df = demo_datasets.load_kansas_test_wells()
        else:
            df = demo_datasets.load_facies_well_data()
        
        # Filter by well if specified
        if well_name and 'Well Name' in df.columns:
            df = df[df['Well Name'] == well_name]
        
        # Create facies log plot
        from geosuite_lib.plotting.strip_charts import create_facies_log_plot
        
        fig = create_facies_log_plot(
            df=df,
            depth_col='Depth',
            facies_col='Facies',
            well_name=well_name
        )
        
        return jsonify({
            "status": "success",
            "figure": fig.to_json(),
            "n_samples": len(df),
            "wells": df['Well Name'].unique().tolist() if 'Well Name' in df.columns else []
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": "Failed to generate facies strip chart",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
