"""
GeoSuite Flask app factory.
Creates and configures the Flask application.
Unified geomechanics and subsurface geology workflows.
"""

import os
from flask import Flask
from .blueprints.main import bp as main_bp
from .blueprints.api import bp as api_bp
from .blueprints.geomech import bp as geomech_bp
from .blueprints.petro import bp as petro_bp
from .blueprints.data import bp as data_bp
from .blueprints.ml import bp as ml_bp
from .blueprints.wells import bp as wells_bp

try:
    from .blueprints.workflows import bp as workflows_bp
    _has_workflows = True
except ImportError:
    _has_workflows = False
    workflows_bp = None


def create_app():
    """Create and configure Flask application.
    
    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Basic configuration
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "geosuite-development-key")
    
    # GeoSuite configuration
    app.config["OFFLINE_MODE"] = os.getenv("OFFLINE_MODE", "true").lower() == "true"
    app.config["DATA_DIRECTORY"] = os.getenv("DATA_DIRECTORY", "geosuite_lib/data")
    
    # Databricks configuration (optional)
    app.config["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST", "")
    app.config["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "")
    app.config["DATABRICKS_WAREHOUSE_ID"] = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
    app.config["DATABRICKS_CATALOG"] = os.getenv("DATABRICKS_CATALOG", "")
    app.config["DATABRICKS_SCHEMA"] = os.getenv("DATABRICKS_SCHEMA", "")
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(geomech_bp, url_prefix="/geomech")
    app.register_blueprint(petro_bp, url_prefix="/petro")
    app.register_blueprint(data_bp, url_prefix="/data")
    app.register_blueprint(ml_bp, url_prefix="/ml")
    app.register_blueprint(wells_bp, url_prefix="/wells")
    
    # Register workflows blueprint if available
    if _has_workflows and workflows_bp:
        app.register_blueprint(workflows_bp, url_prefix="/workflows")
    
    # Register REST API and Swagger if available
    try:
        from .blueprints.api.rest_api import bp as rest_api_bp
        from .blueprints.api.swagger import bp as swagger_bp
        app.register_blueprint(rest_api_bp)
        app.register_blueprint(swagger_bp)
    except ImportError:
        pass
    
    return app
