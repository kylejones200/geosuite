"""
Main blueprint for GeoSuite home page and navigation.
"""

from flask import Blueprint

bp = Blueprint('main', __name__)

from . import routes
