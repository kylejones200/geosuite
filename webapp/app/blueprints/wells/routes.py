"""
Wells blueprint routes for North Dakota wells DCA functionality.
"""

import logging
from flask import jsonify, render_template, request
from . import bp
from ...services.databricks_service import query_wells_geojson, query_operators_data, query_well_detail

logger = logging.getLogger(__name__)


@bp.route("/")
def wells_home():
    """Render the North Dakota Wells DCA application."""
    return render_template("wells/index.html")


@bp.route("/api/wells")
def api_wells():
    """Return wells data from Delta tables in GeoJSON format"""
    try:
        limit = request.args.get('limit', 1000, type=int)
        wells_geojson = query_wells_geojson(limit)
        logger.info(f"Returning {len(wells_geojson.get('features', []))} wells")
        return jsonify(wells_geojson)
    except Exception as e:
        logger.error(f"Error in /api/wells: {e}")
        return jsonify({'error': 'Failed to load wells data', 'details': str(e)}), 500


@bp.route("/api/operators")
def api_operators():
    """Return operators performance data from Delta tables"""
    try:
        since = request.args.get('since')
        operators_data = query_operators_data(since)
        logger.info(f"Returning {len(operators_data.get('operators', []))} operators")
        return jsonify(operators_data)
    except Exception as e:
        logger.error(f"Error in /api/operators: {e}")
        return jsonify({'error': 'Failed to load operators data', 'details': str(e)}), 500


@bp.route("/api/wells/<uwi>")
def api_well_detail(uwi):
    """Return detailed data for a specific well"""
    try:
        well_data = query_well_detail(uwi)
        logger.info(f"Returning detail for well: {uwi}")
        return jsonify(well_data)
    except Exception as e:
        logger.error(f"Error in /api/wells/{uwi}: {e}")
        return jsonify({'error': 'Failed to load well detail', 'details': str(e)}), 500


@bp.route("/health")
def wells_health():
    """Wells module health check."""
    return jsonify({
        "status": "healthy",
        "module": "North Dakota Wells DCA",
        "version": "1.0"
    })

