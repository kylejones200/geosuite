"""
Wells blueprint for North Dakota wells DCA functionality.
"""

from flask import Blueprint

bp = Blueprint('wells', __name__)

from . import routes

