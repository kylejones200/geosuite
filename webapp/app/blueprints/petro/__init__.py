"""
Petrophysics blueprint for reservoir analysis and log calculations.
"""

from flask import Blueprint

bp = Blueprint('petro', __name__)

from . import routes
