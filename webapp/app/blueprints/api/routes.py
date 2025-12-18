"""
API blueprint routes for data endpoints.
"""

from flask import jsonify, request, current_app
from . import bp
import os
import pandas as pd


@bp.route("/health")
def health():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "api": "GeoSuite API",
        "version": "1.0"
    })


@bp.route("/data/demo")
def get_demo_data():
    """Get demo dataset for testing."""
    try:
        data_dir = current_app.config.get("DATA_DIRECTORY", "geosuite_lib/data")
        demo_file = os.path.join(data_dir, "files", "demo_well_logs.csv")
        
        if os.path.exists(demo_file):
            df = pd.read_csv(demo_file)
            return jsonify({
                "status": "success",
                "data": df.head(100).to_dict(orient="records"),
                "total_rows": len(df),
                "columns": list(df.columns)
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Demo data file not found",
                "file": demo_file
            }), 404
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to load demo data",
            "error": str(e)
        }), 500


@bp.route("/data/upload", methods=["POST"])
def upload_data():
    """Upload and process data files."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Process file based on extension
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
            return jsonify({
                "status": "success",
                "message": f"CSV file processed successfully",
                "rows": len(df),
                "columns": list(df.columns),
                "sample_data": df.head(5).to_dict(orient="records")
            })
        elif filename.endswith('.las'):
            return jsonify({
                "status": "partial",
                "message": "LAS file upload detected - processing not yet implemented",
                "filename": filename
            })
        else:
            return jsonify({
                "error": "Unsupported file type",
                "supported": [".csv", ".las"]
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to process uploaded file",
            "error": str(e)
        }), 500
