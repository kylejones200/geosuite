"""
Data blueprint for file import and management.
"""

from flask import Blueprint

bp = Blueprint('data', __name__)

from . import routes
