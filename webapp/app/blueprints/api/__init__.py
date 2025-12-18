"""
API blueprint for GeoSuite data endpoints.
"""

from flask import Blueprint

bp = Blueprint('api', __name__)

from . import routes
