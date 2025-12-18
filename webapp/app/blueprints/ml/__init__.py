"""
ML blueprint for machine learning model management with MLflow.
"""

from flask import Blueprint

bp = Blueprint('ml', __name__)

from . import routes
