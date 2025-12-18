"""
Main blueprint routes for GeoSuite home page.
"""

from flask import render_template, current_app
from . import bp


@bp.route("/")
def index():
    """Render the GeoSuite home page."""
    return render_template("index.html")


@bp.route("/health")
def health():
    """Health check endpoint.
    
    Returns:
        dict: Health status information.
    """
    return {
        "status": "healthy",
        "app": "GeoSuite",
        "offline_mode": current_app.config.get("OFFLINE_MODE", True),
        "data_directory": current_app.config.get("DATA_DIRECTORY", "geosuite_lib/data")
    }
