"""
Geomechanics blueprint for stress calculations and wellbore stability.
"""

from flask import Blueprint

bp = Blueprint('geomech', __name__)

from . import routes
